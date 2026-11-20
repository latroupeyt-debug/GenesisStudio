"""Boucle d'entraînement : reprise depuis une version précédente, checkpoints,
lecture du corpus en mémoire virtuelle (RAM constante).

Le principe de reprise :
    - une nouvelle version peut démarrer depuis les *poids* d'une version
      existante (transfert d'apprentissage) ;
    - une version interrompue peut reprendre exactement là où elle s'est
      arrêtée (poids + optimiseur + compteurs), sans rien recalculer.
"""

from __future__ import annotations

import json
import math
import queue
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import gpu
from .model import GPT, ModelConfig, device_label, pick_device


@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 16
    seq_len: int = 128
    lr: float = 1e-3
    warmup: int = 40
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    grad_accum: int = 1
    eval_every: int = 50
    ckpt_every: int = 100
    seed: int = 1337
    threads: int = 0
    device: str = "auto"
    max_steps: int = 0
    amp: bool = True          # précision mixte automatique sur GPU
    compile_model: bool = False  # torch.compile expérimental, repli eager automatique
    prefetch: int = 2         # lots préparés à l'avance (masque la latence CPU→GPU)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in (data or {}).items() if k in known})

    @property
    def tokens_per_step(self) -> int:
        return max(1, self.batch_size * self.seq_len * max(1, self.grad_accum))


# ---------------------------------------------------------------------------
# Jeu de données : fenêtres aléatoires lues dans le fichier de tokens
# ---------------------------------------------------------------------------
class TokenWindows:
    """Échantillonne des fenêtres (x, y) sans jamais charger le fichier entier."""

    def __init__(self, token_path: Path, dtype: str, n_tokens: int,
                 seq_len: int, seed: int = 1337, holdout: float = 0.02):
        self.path = Path(token_path)
        self.dtype = dtype
        self.data = np.memmap(str(self.path), dtype=dtype, mode="r")
        self.total = min(int(n_tokens), int(self.data.shape[0]))
        if self.total < seq_len + 8:
            raise ValueError(
                f"Le corpus ne contient que {self.total:,} tokens alors que la longueur de "
                f"contexte demandée est de {seq_len}. Ajoutez des textes ou réduisez le contexte."
                .replace(",", " ")
            )
        reserve = max(1024, int(self.total * holdout))
        reserve = min(reserve, max(0, self.total // 4))
        self.n_train = max(seq_len + 2, self.total - reserve)
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def batch(self, batch_size: int, device: torch.device, pin: bool = False):
        starts = self.rng.integers(0, self.n_train - self.seq_len - 1, size=batch_size)
        chunk = np.stack([
            np.asarray(self.data[s : s + self.seq_len + 1], dtype=np.int64) for s in starts
        ])
        tensor = torch.from_numpy(chunk)
        x = tensor[:, :-1]
        y = tensor[:, 1:]
        if pin:
            # mémoire épinglée : le transfert vers le GPU devient asynchrone
            x, y = x.pin_memory(), y.pin_memory()
            return x, y
        return x.to(device), y.to(device)

    def eval_batches(self, batch_size: int, device: torch.device, count: int = 8):
        start = self.n_train
        span = self.total - start - 1
        if span < self.seq_len + 1:
            return
        step = max(self.seq_len, span // count)
        for i in range(count):
            s = start + i * step
            if s + self.seq_len + 1 > self.total:
                break
            chunk = torch.from_numpy(
                np.asarray(self.data[s : s + self.seq_len + 1], dtype=np.int64)
            ).unsqueeze(0)
            yield chunk[:, :-1].to(device), chunk[:, 1:].to(device)


class BatchFeeder(threading.Thread):
    """Prépare les lots à l'avance dans un fil dédié.

    Le passage mémoire virtuelle → tenseur → (épinglé) → GPU est ainsi recouvert par
    le calcul du pas précédent. Sur GPU, cela évite que le CPU devienne le goulot.
    """

    def __init__(self, data: "TokenWindows", train_cfg: "TrainConfig",
                 device: torch.device, size: int = 2):
        super().__init__(daemon=True)
        self.data = data
        self.cfg = train_cfg
        self.device = device
        self.pin = device.type == "cuda"
        self.queue: queue.Queue = queue.Queue(maxsize=max(1, size))
        self._stop_event = threading.Event()
        self._error: Exception | None = None

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                x, y = self.data.batch(self.cfg.batch_size, self.device, pin=self.pin)
            except Exception as exc:  # remonte l'erreur au consommateur
                self._error = exc
                return
            while not self._stop_event.is_set():
                try:
                    self.queue.put((x, y), timeout=0.2)
                    break
                except queue.Full:
                    continue
            else:
                return

    def get(self):
        # Le préchargement CPU n'apporte pas de transfert à masquer : rester synchrone
        # évite un fil de NumPy/Torch inutile sur les machines sans accélérateur.
        if not self.pin:
            return self.data.batch(self.cfg.batch_size, self.device, pin=False)
        while True:
            try:
                item = self.queue.get(timeout=0.2)
                break
            except queue.Empty:
                if not self.is_alive():
                    detail = f": {self._error}" if self._error else "."
                    raise RuntimeError("Préparation des lots interrompue" + detail)
        if item is None:
            raise RuntimeError("Préparation des lots interrompue.")
        x, y = item
        if self.pin:
            return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
        return x, y

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------
def _unwrap_compiled(model):
    """Retire les wrappers OptimizedModule pour garder les checkpoints portables."""
    while hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def save_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer | None,
                    meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = _unwrap_compiled(model)
    payload = {
        "format": 1,
        "model_config": model.cfg.to_dict(),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": meta,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def read_checkpoint_meta(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return payload.get("meta", {}) | {"model_config": payload.get("model_config", {})}
    except Exception:
        return {}


def checkpoint_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 ** 2)
    except OSError:
        return 0.0


# ---------------------------------------------------------------------------
# Entraîneur
# ---------------------------------------------------------------------------
@dataclass
class TrainHooks:
    on_log: Callable[[str], None] | None = None
    on_step: Callable[[dict], None] | None = None
    on_checkpoint: Callable[[dict], None] | None = None


class Trainer:
    def __init__(
        self,
        *,
        token_path: Path,
        dtype: str,
        n_tokens: int,
        model: GPT,
        train_cfg: TrainConfig,
        ckpt_last: Path,
        ckpt_best: Path,
        initial_step: int = 0,
        initial_tokens_seen: int = 0,
        initial_history: list | None = None,
        best_loss: float | None = None,
        optimizer_state: dict | None = None,
        hooks: TrainHooks | None = None,
    ):
        self.token_path = Path(token_path)
        self.dtype = dtype
        self.n_tokens = n_tokens
        self.model = model
        self.cfg = train_cfg
        self.ckpt_last = Path(ckpt_last)
        self.ckpt_best = Path(ckpt_best)
        self.hooks = hooks or TrainHooks()
        self.device = pick_device(train_cfg.device)
        self.data = TokenWindows(self.token_path, self.dtype, n_tokens,
                                 train_cfg.seq_len, train_cfg.seed)
        self.steps_per_epoch = max(1, self.data.n_train // self.cfg.tokens_per_step)
        total = self.steps_per_epoch * max(1, self.cfg.epochs)
        if self.cfg.max_steps:
            total = min(total, self.cfg.max_steps)
        self.total_steps = int(total)
        self.start_step = int(initial_step)
        self.tokens_seen = int(initial_tokens_seen)
        self.history: list = list(initial_history or [])
        self.best_loss = best_loss
        self.optimizer_state = optimizer_state
        self._stop = threading.Event()
        self.step = self.start_step
        self.last_loss: float | None = None
        self.started_at = 0.0

    # -- contrôle -------------------------------------------------------
    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def log(self, message: str) -> None:
        if self.hooks.on_log:
            self.hooks.on_log(message)

    # -- programme de learning rate -------------------------------------
    def lr_at(self, step: int) -> float:
        lr, warmup = self.cfg.lr, max(1, self.cfg.warmup)
        if step < warmup:
            return lr * (step + 1) / warmup
        span = max(1, self.total_steps - warmup)
        ratio = min(1.0, (step - warmup) / span)
        min_lr = lr * self.cfg.min_lr_ratio
        return min_lr + (lr - min_lr) * 0.5 * (1 + math.cos(math.pi * ratio))

    # -- exécution ------------------------------------------------------
    def run(self) -> dict:
        cfg = self.cfg
        if cfg.threads and cfg.threads > 0:
            torch.set_num_threads(cfg.threads)
        torch.manual_seed(cfg.seed + self.start_step)
        gpu.configure_backends(self.device)

        amp_type = gpu.amp_dtype(self.device) if cfg.amp else None
        self.use_amp = amp_type is not None
        self.amp_type = amp_type or torch.float32
        self.scaler = gpu.make_grad_scaler(
            enabled=(self.use_amp and self.amp_type is torch.float16)
        )
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        base_model = self.model.to(self.device)
        model = base_model
        optimizer = base_model.configure_optimizer(cfg.lr, cfg.weight_decay)
        compile_pending = False
        if cfg.compile_model:
            compiler = getattr(torch, "compile", None)
            if compiler is None:
                self.log("torch.compile n'est pas disponible dans cette version de PyTorch ; mode eager.")
            else:
                try:
                    model = compiler(base_model)
                    compile_pending = True
                    self.log("torch.compile activé : compilation à la première passe, démarrage potentiellement plus lent.")
                except Exception as exc:
                    model = base_model
                    self.log(f"Compilation indisponible ({exc}) ; repli automatique en mode standard.")
        if self.optimizer_state:
            try:
                optimizer.load_state_dict(self.optimizer_state)
            except (ValueError, KeyError):
                self.log("État de l'optimiseur incompatible : réinitialisé.")
        for group in optimizer.param_groups:
            group["lr"] = self.lr_at(self.start_step)

        self.started_at = time.time()
        precision = ""
        if self.use_amp:
            precision = (" · précision mixte bfloat16" if self.amp_type is torch.bfloat16
                         else " · précision mixte float16")
        self.log(
            f"Démarrage — {device_label(self.device)}{precision} · "
            f"{base_model.num_parameters():,} paramètres · "
            f"{self.data.n_train:,} tokens d'entraînement · "
            f"{self.steps_per_epoch:,} pas/époque · total {self.total_steps:,} pas"
        )
        if self.device.type == "cuda":
            self.log(
                f"VRAM : {gpu.vram_gb():.1f} Go sur « {gpu.device_report().get('name')} » · "
                f"lot de {cfg.batch_size} × {cfg.seq_len} tokens"
            )
        if self.start_step:
            self.log(f"Reprise au pas {self.start_step:,} ({self.tokens_seen:,} tokens déjà vus).")

        feeder = BatchFeeder(self.data, cfg, self.device, size=max(1, cfg.prefetch))
        if feeder.pin:
            feeder.start()
        tokens_at_start = self.tokens_seen
        step_time = 0.0
        interrupted = False
        # points de courbe : au moins ~250 points, même sans évaluation
        history_every = max(1, self.total_steps // 250)

        try:
            for step in range(self.start_step, self.total_steps):
                if self.stopped:
                    interrupted = True
                    break
                t0 = time.perf_counter()
                lr = self.lr_at(step)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                batches = [feeder.get() for _ in range(max(1, cfg.grad_accum))]

                def compute_gradients(active_model):
                    active_model.train()
                    optimizer.zero_grad(set_to_none=True)
                    accumulated = 0.0
                    for x, y in batches:
                        with torch.autocast(device_type=self.device.type, dtype=self.amp_type,
                                            enabled=self.use_amp):
                            _, loss = active_model(x, y)
                        scaled = loss / max(1, cfg.grad_accum)
                        if self.scaler.is_enabled():
                            self.scaler.scale(scaled).backward()
                        else:
                            scaled.backward()
                        accumulated += float(loss.detach())
                    return accumulated / max(1, cfg.grad_accum)

                try:
                    accum_loss = compute_gradients(model)
                    if compile_pending:
                        self.log("torch.compile opérationnel.")
                        compile_pending = False
                except Exception as exc:
                    # Les backends de compilation peuvent manquer (ex. Triton sous Windows).
                    # Ne consomme pas un second lot et n'applique aucun pas partiel : on
                    # recalcule les mêmes fenêtres en mode eager avant optimizer.step().
                    if not compile_pending:
                        raise
                    optimizer.zero_grad(set_to_none=True)
                    if self.device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    model = base_model
                    compile_pending = False
                    self.log(f"Compilation échouée au premier pas ({exc}) ; reprise en mode standard.")
                    accum_loss = compute_gradients(base_model)

                if cfg.grad_clip:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), cfg.grad_clip)
                if self.scaler.is_enabled():
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    optimizer.step()

                self.step = step + 1
                self.tokens_seen += cfg.tokens_per_step
                train_loss = accum_loss / max(1, cfg.grad_accum)
                self.last_loss = train_loss
                step_time = time.perf_counter() - t0

                need_eval = cfg.eval_every > 0 and (self.step % cfg.eval_every == 0 or self.step == self.total_steps)
                eval_loss = self._evaluate(model) if need_eval else None
                if need_eval:
                    if eval_loss is not None and (self.best_loss is None or eval_loss < self.best_loss):
                        self.best_loss = eval_loss
                        save_checkpoint(self.ckpt_best, model, optimizer, self._meta())
                if self.step % history_every == 0 or need_eval or self.step == self.total_steps:
                    self.history.append([self.step, round(train_loss, 5),
                                         None if eval_loss is None else round(eval_loss, 5)])
                    self._trim_history()

                if self.hooks.on_step:
                    elapsed = max(1e-6, time.time() - self.started_at)
                    done_steps = max(1, self.step - self.start_step)
                    per_step = elapsed / done_steps
                    remaining = max(0, self.total_steps - self.step)
                    self.hooks.on_step({
                        "step": self.step,
                        "total_steps": self.total_steps,
                        "loss": train_loss,
                        "eval_loss": eval_loss,
                        "lr": lr,
                        "tokens_seen": self.tokens_seen,
                        "tokens_new": self.tokens_seen - tokens_at_start,
                        "tok_per_s": cfg.tokens_per_step / max(per_step, 1e-6),
                        "eta": remaining * per_step,
                        "progress": self.step / max(1, self.total_steps),
                        "epoch": self.step / self.steps_per_epoch,
                    })

                if cfg.ckpt_every > 0 and self.step % cfg.ckpt_every == 0:
                    self._save_last(model, optimizer)
                    if self.hooks.on_checkpoint:
                        self.hooks.on_checkpoint(self._meta())

        finally:
            if feeder.is_alive():
                feeder.stop()
                feeder.join(timeout=2.0)
        if interrupted:
            self.log("Arrêt demandé : sauvegarde du dernier état…")
        self._save_last(model, optimizer)
        if self.hooks.on_checkpoint:
            self.hooks.on_checkpoint(self._meta())

        duration = time.time() - self.started_at
        summary = {
            "steps": self.step,
            "tokens_seen": self.tokens_seen,
            "loss": self.last_loss,
            "best_loss": self.best_loss,
            "history": self.history,
            "duration": duration,
            "interrupted": interrupted,
            "total_steps": self.total_steps,
            "device": self.device.type,
        }
        self.log(
            f"{'Entraînement interrompu' if interrupted else 'Entraînement terminé'} — "
            f"{self.step - self.start_step:,} pas effectués en "
            f"{duration / 60:.1f} min · perte finale {self._fmt(self.last_loss)}"
        )
        return summary

    # -- utilitaires ----------------------------------------------------
    @staticmethod
    def _fmt(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    @torch.no_grad()
    def _evaluate(self, model: GPT) -> float | None:
        model.eval()
        losses = []
        for x, y in self.data.eval_batches(self.cfg.batch_size, self.device, count=6):
            with torch.autocast(device_type=self.device.type, dtype=self.amp_type,
                                enabled=self.use_amp):
                _, loss = model(x, y)
            losses.append(float(loss))
        model.train()
        return float(np.mean(losses)) if losses else None

    def _trim_history(self) -> None:
        if len(self.history) > 6000:
            self.history = self.history[-6000:]

    def _meta(self) -> dict:
        return {
            "step": self.step,
            "total_steps": self.total_steps,
            "steps_per_epoch": self.steps_per_epoch,
            "tokens_seen": self.tokens_seen,
            "loss": self.last_loss,
            "best_loss": self.best_loss,
            "history": self.history,
            "train_config": self.cfg.to_dict(),
            "device": self.device.type,
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    def _save_last(self, model: GPT, optimizer: torch.optim.Optimizer) -> None:
        save_checkpoint(self.ckpt_last, model, optimizer, self._meta())


# ---------------------------------------------------------------------------
# Aides hors boucle (utilisées par l'interface)
# ---------------------------------------------------------------------------
def build_model(model_cfg: dict) -> GPT:
    return GPT(ModelConfig.from_dict(model_cfg))


def warm_start(model: GPT, checkpoint_path: Path, device: torch.device) -> tuple[bool, str]:
    """Charge les poids d'une version antérieure dans le nouveau modèle."""
    if not Path(checkpoint_path).exists():
        return False, "Aucun checkpoint trouvé dans la version de départ."
    try:
        payload = load_checkpoint(Path(checkpoint_path), device)
    except Exception as exc:
        return False, f"Checkpoint illisible : {exc}"
    state = payload.get("model_state", {})
    old_cfg = payload.get("model_config", {})
    new_cfg = model.cfg.to_dict()
    mismatched = [k for k in ("vocab_size", "block_size", "n_embd", "n_layer", "n_head")
                  if old_cfg.get(k) != new_cfg.get(k)]
    if mismatched:
        return False, ("Architecture différente de la version de départ "
                       f"({', '.join(mismatched)}) : poids non transférés.")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        return False, f"Poids incompatibles : {exc}"
    return True, f"Poids de la version de départ chargés ({len(state)} tenseurs)."


def generation_engine(model: GPT, device: torch.device):
    model = model.to(device)
    model.eval()
    return model
