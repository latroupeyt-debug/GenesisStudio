"""Détection et exploitation du GPU (CUDA / MPS), et calibration des réglages.

Le code d'entraînement utilise ce module pour :
    - savoir ce que la machine a réellement comme GPU (nom, VRAM, version CUDA) ;
    - activer TF32 / la précision mixte quand c'est pertinent ;
    - proposer une taille de lot réaliste, mesurée sur la carte (calibration).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .model import GPT, ModelConfig, device_label, human_params


# ---------------------------------------------------------------------------
# Informations sur la machine
# ---------------------------------------------------------------------------
def cuda_build() -> bool:
    """Vrai si la version de PyTorch installée a été compilée avec CUDA."""
    return bool(getattr(torch.version, "cuda", None))


def cuda_available() -> bool:
    return bool(torch.cuda.is_available())


def gpu_count() -> int:
    return torch.cuda.device_count() if cuda_available() else 0


def bf16_supported() -> bool:
    if not cuda_available():
        return False
    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def vram_gb(index: int = 0) -> float:
    if not cuda_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(index)
        return props.total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def device_report(index: int = 0) -> dict:
    """Instantané lisible de l'état matériel/logiciel."""
    report = {
        "torch": torch.__version__,
        "cuda_build": cuda_build(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "available": cuda_available(),
        "count": gpu_count(),
        "name": None,
        "vram_gb": 0.0,
        "vram_free_gb": 0.0,
        "capability": None,
        "bf16": bf16_supported(),
        "sdpa": hasattr(torch.nn.functional, "scaled_dot_product_attention"),
        "flash_sdp": False,
        "torch_compile": callable(getattr(torch, "compile", None)),
        "tf32": None,
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "label": device_label(torch.device("cpu")),
    }
    if cuda_available():
        try:
            props = torch.cuda.get_device_properties(index)
            report.update({
                "name": props.name,
                "vram_gb": props.total_memory / (1024 ** 3),
                "capability": f"{props.major}.{props.minor}",
                "label": device_label(torch.device("cuda")),
            })
            free, _total = torch.cuda.mem_get_info(index)
            report["vram_free_gb"] = free / (1024 ** 3)
        except Exception:
            pass
    try:
        report["flash_sdp"] = bool(torch.backends.cuda.flash_sdp_enabled())
    except Exception:
        pass
    try:
        report["tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    except Exception:
        report["tf32"] = None
    return report


def configure_backends(device: torch.device) -> None:
    """Active les accélérateurs sûrs pour le matériel détecté."""
    if device.type == "cuda":
        # TF32 : gain notable sur les GPU Ampere+ avec une perte de précision minime
        try:
            if torch.cuda.get_device_capability(0)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True


def amp_dtype(device: torch.device) -> torch.dtype | None:
    """Type de la précision mixte, ou None si non pertinent."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if bf16_supported() else torch.float16


def make_grad_scaler(enabled: bool):
    """Crée le scaler AMP sur PyTorch récent et ancien (>= 2.1)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # API historique disponible dans torch.cuda.amp
        return torch.cuda.amp.GradScaler(enabled=enabled)


def install_hint() -> str:
    """Commande à donner à l'utilisateur pour activer CUDA."""
    return ("pip install torch --index-url https://download.pytorch.org/whl/cu126  "
            "(adaptez cu126 à votre version de pilote NVIDIA)")


# ---------------------------------------------------------------------------
# Estimation mémoire
# ---------------------------------------------------------------------------
def estimate_weights_mb(cfg: ModelConfig, bytes_per_param: float = 12.0) -> float:
    """Poids + états AdamW (2 moments) ~ 12 octets par paramètre en float32."""
    from .model import count_parameters

    return count_parameters(cfg) * bytes_per_param / (1024 ** 2)


def estimate_activation_mb(cfg: ModelConfig, batch: int, seq_len: int) -> float:
    """Estimation très grossière des activations par pas avant (en Mo)."""
    # ~ 6 tenseurs de taille (batch, seq, d_model) par couche + attention (batch, têtes, seq, seq)
    per_layer = 6 * cfg.n_embd + cfg.n_head * seq_len
    return batch * seq_len * cfg.n_layer * per_layer * 4 / (1024 ** 2)


def suggested_batch(cfg: ModelConfig, seq_len: int, vram_total_gb: float,
                    reserve: float = 0.25) -> int:
    """Taille de lot prudente calculée à partir de la VRAM (avant calibration)."""
    if vram_total_gb <= 0:
        return 8
    weights = estimate_weights_mb(cfg) / 1024
    usable = max(0.2, vram_total_gb * (1 - reserve) - weights * 1.6)
    per_sample = max(0.002, estimate_activation_mb(cfg, 1, seq_len) / 1024 * 2.4)
    batch = int(usable / per_sample)
    for candidate in (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128):
        if candidate >= batch:
            return candidate
    return 128


# ---------------------------------------------------------------------------
# Calibration : on mesure au lieu de deviner
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkResult:
    device: str = "cpu"
    device_name: str = ""
    params: int = 0
    seq_len: int = 128
    batch: int = 0
    max_batch: int = 0
    tokens_per_s: float = 0.0
    step_ms: float = 0.0
    amp: bool = False
    amp_dtype: str = ""
    error: str = ""
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.error:
            return self.error
        bits = [
            f"{self.device_name or self.device} · {human_params(self.params)} paramètres",
            f"lot validé : {self.max_batch}",
            f"débit : {self.tokens_per_s:,.0f} tokens/s".replace(",", " "),
        ]
        if self.amp:
            bits.append(f"précision mixte {self.amp_dtype}")
        return " · ".join(bits)


def run_benchmark(
    model_cfg: dict,
    device_pref: str = "auto",
    seq_len: int = 128,
    max_batch: int = 128,
    max_seconds: float = 20.0,
    progress=None,
    should_stop=None,
) -> BenchmarkResult:
    """Mesure le lot maximal qui tient en mémoire et le débit réel.

    La séquence est volontairement courte : on cherche un ordre de grandeur utile
    pour régler la taille de lot, pas un score de référence.
    """
    from .model import pick_device

    cfg = ModelConfig.from_dict(model_cfg)
    cfg.validate()
    seq_len = max(8, min(int(seq_len), cfg.block_size))
    device = pick_device(device_pref)
    result = BenchmarkResult(device=device.type, seq_len=seq_len, params=0)
    result.device_name = device_label(device)

    from .model import count_parameters

    result.params = count_parameters(cfg)
    opaque = cfg.dropout
    cfg.dropout = 0.0

    try:
        model = GPT(cfg).to(device)
    except RuntimeError as exc:
        result.error = f"Impossible de créer le modèle sur {device.type} : {exc}"
        cfg.dropout = opaque
        return result
    model.train()
    optimizer = model.configure_optimizer(1e-4, 0.0)

    dtype = amp_dtype(device)
    use_amp = dtype is not None
    scaler = make_grad_scaler(enabled=(use_amp and dtype is torch.float16))
    if use_amp:
        result.amp = True
        result.amp_dtype = "bfloat16" if dtype is torch.bfloat16 else "float16"

    def one_step(batch: int, measure: bool = False) -> float:
        x = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
        y = torch.randint(0, cfg.vocab_size, (batch, seq_len), device=device)
        started = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=dtype or torch.float32,
                            enabled=use_amp):
            _logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started

    # 1) montée en taille de lot jusqu'à l'échec mémoire
    batch = 1
    best = 0
    history: list[tuple[int, float]] = []
    while batch <= max_batch:
        if should_stop and should_stop():
            break
        try:
            if progress:
                progress(0.2 + 0.6 * min(1.0, batch / max_batch), f"lot {batch}…")
            one_step(batch)  # échauffement (allocation)
            elapsed = min(one_step(batch), 5.0) if batch <= 32 else one_step(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            history.append((batch, elapsed))
            best = batch
            if elapsed > max_seconds / 4:
                break
            batch *= 2
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                result.notes.append(f"lot {batch} : mémoire insuffisante")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
            result.error = str(exc)
            break

    result.max_batch = best
    if best >= max_batch:
        result.notes.append(
            f"Plafond du test atteint ({max_batch}) : le maximum réel peut être supérieur."
        )
    if best:
        # 2) débit mesuré sur un lot intermédiaire (plus représentatif)
        probe = max(1, min(best, 8))
        timings = []
        for _ in range(3):
            try:
                timings.append(one_step(probe, measure=True))
            except RuntimeError:
                break
        if timings:
            step_s = sorted(timings)[len(timings) // 2]
            result.batch = probe
            result.step_ms = step_s * 1000
            result.tokens_per_s = (probe * seq_len) / max(step_s, 1e-6)
        if device.type == "cuda":
            result.notes.append(
                f"VRAM utilisée au pic : {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} Go"
            )

    # nettoyage
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    cfg.dropout = opaque
    return result
