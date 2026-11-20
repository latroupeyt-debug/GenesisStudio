"""Orchestration de haut niveau : tokenisation du corpus, création de version,
transfert des poids depuis une version précédente, reprise exacte.

L'interface ne parle qu'à ce module (et à `project`), ce qui garde l'UI simple et
rend le moteur réutilisable (CLI, tests, futures fonctions).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import torch

from . import corpus
from .model import (
    GPT,
    ModelConfig,
    count_parameters,
    human_params,
    pick_device,
    preset_model_config,
)
from .project import Project, Version
from .tokenizers import Tokenizer, create_tokenizer
from .trainer import (
    TrainConfig,
    Trainer,
    TrainHooks,
    load_checkpoint,
    read_checkpoint_meta,
    save_checkpoint,
    warm_start,
)

LAST_CKPT = "last.pt"
BEST_CKPT = "best.pt"
INIT_CKPT = "init.pt"


# ---------------------------------------------------------------------------
# 1. Tokenisation du corpus (obligatoire avant tout entraînement)
# ---------------------------------------------------------------------------
def tokenize_project(
    project: Project,
    progress: Callable[[float, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    files = project.corpus_files
    if not files:
        raise ValueError("Le corpus est vide : ajoutez au moins un fichier texte.")

    info = project.tokenizer_info
    kind = info.get("kind", "char")
    options = dict(info.get("options", {}))

    def fit_progress(fraction: float, message: str) -> None:
        if progress:
            progress(fraction * 0.35, f"Apprentissage du tokenizer — {message}")

    tokenizer = create_tokenizer(kind)
    # toujours ré-entraîner le tokenizer sur le corpus courant (le vocabulaire
    # doit correspondre au corpus, et cela garantit la reproductibilité)
    chunks = corpus.iter_corpus_text(files, project.corpus_dir, add_separators=False)
    tokenizer.fit(chunks, progress=fit_progress, should_stop=should_stop, **options)
    if should_stop and should_stop():
        raise InterruptedError("Tokenisation annulée.")
    project.set_tokenizer(tokenizer)

    stats = corpus.corpus_stats(files)
    t0 = time.time()

    def build_progress(fraction: float, message: str) -> None:
        if progress:
            progress(0.35 + fraction * 0.65, f"Encodage du corpus — {message}")

    result = corpus.build_token_file(
        files,
        project.corpus_dir,
        tokenizer,
        project.token_file,
        total_chars=stats["chars"],
        progress=build_progress,
        should_stop=should_stop,
    )
    if should_stop and should_stop():
        raise InterruptedError("Tokenisation annulée.")

    result.update({
        "signature": project.corpus_signature(),
        "kind": kind,
        "vocab_size": tokenizer.vocab_size,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": round(time.time() - t0, 1),
    })
    project.data["tokens"] = result
    project.save()
    return result


def ensure_tokens(project: Project) -> bool:
    return project.tokens_ready


# ---------------------------------------------------------------------------
# 2. Création d'une version (avec ou sans poids hérités)
# ---------------------------------------------------------------------------
def _parent_checkpoint(project: Project, parent_id: str) -> Path | None:
    for name in (LAST_CKPT, BEST_CKPT, INIT_CKPT):
        path = project.ckpt_dir(parent_id) / name
        if path.exists():
            return path
    return None


def build_version(
    project: Project,
    *,
    preset_key: str,
    model_overrides: dict | None = None,
    train_cfg: TrainConfig,
    base_version_id: str | None = None,
    name: str | None = None,
    transfer_weights: bool = True,
    device_pref: str = "auto",
) -> tuple[Version, str, bool]:
    """Crée une version et, si demandé, y transfère les poids d'une version existante.

    Renvoie (version, message, poids_transférés).
    """
    if not project.tokens_ready:
        raise ValueError("Le corpus doit d'abord être analysé (onglet Corpus).")

    vocab_size = int(project.tokenizer_info.get("vocab_size", 0))
    parent = project.version(base_version_id) if base_version_id else None

    if parent is not None and transfer_weights:
        # architecture verrouillée : on garde exactement la même structure
        model_cfg = dict(parent.model)
        model_cfg["vocab_size"] = vocab_size
        model_cfg["preset"] = parent.model.get("preset", preset_key)
        model_cfg["preset_label"] = parent.model.get("preset_label", "héritée")
    else:
        model_cfg = preset_model_config(preset_key, vocab_size)
        if model_overrides:
            model_cfg.update({k: v for k, v in model_overrides.items() if v is not None})
        ModelConfig.from_dict(model_cfg).validate()

    model_cfg["params"] = count_parameters(ModelConfig.from_dict(model_cfg))
    train_dict = train_cfg.to_dict()
    version = project.create_version(model_cfg, train_dict, parent=parent.id if parent else None, name=name)

    model = GPT(ModelConfig.from_dict(model_cfg))
    message = "Nouveau modèle initialisé de zéro."
    transferred = False
    device = pick_device(device_pref)

    if parent is not None and transfer_weights:
        ckpt = _parent_checkpoint(project, parent.id)
        if ckpt is None:
            message = ("La version de départ n'a pas encore de poids : le modèle démarre de zéro.")
        else:
            ok, msg = warm_start(model, ckpt, device)
            transferred = ok
            message = f"Poids hérités de {parent.name} — {msg}"
            if ok:
                version.note = f"Poids transférés depuis {parent.name}"
    version.tokens_seen = parent.tokens_seen if (parent and transferred) else 0
    if parent is not None and transferred:
        # On ne duplique pas les poids : la version pointe vers le checkpoint parent.
        # (économie d'espace disque, et la reprise reste déterministe)
        ref = _parent_checkpoint(project, parent.id)
        version.init_ref = {
            "version_id": parent.id,
            "checkpoint": Path(ref).name if ref else LAST_CKPT,
            "tokens_seen": parent.tokens_seen,
        }
    project.update_version(version)
    return version, message, transferred


# ---------------------------------------------------------------------------
# 3. Préparation de l'entraînement
# ---------------------------------------------------------------------------
def make_trainer(
    project: Project,
    version: Version,
    hooks: TrainHooks,
    train_cfg: TrainConfig | None = None,
    resume_exact: bool = False,
) -> Trainer:
    """Construit l'entraîneur. `resume_exact` reprend poids + optimiseur + pas."""
    cfg = train_cfg or TrainConfig.from_dict(version.train)
    device = pick_device(cfg.device)
    model = GPT(ModelConfig.from_dict(version.model))

    tokens_meta = project.tokens_meta
    n_tokens = int(tokens_meta.get("tokens", 0))
    dtype = tokens_meta.get("dtype", "uint16")

    initial_step = 0
    tokens_seen = version.tokens_seen
    history: list = list(version.history or [])
    best_loss = version.best_loss
    optimizer_state = None

    ckpt_last = project.ckpt_dir(version.id) / LAST_CKPT
    if resume_exact and ckpt_last.exists():
        payload = load_checkpoint(ckpt_last, device)
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer_state = payload.get("optimizer_state")
        meta = payload.get("meta", {})
        initial_step = int(meta.get("step", version.steps))
        tokens_seen = int(meta.get("tokens_seen", version.tokens_seen))
        history = meta.get("history", history) or history
        best_loss = meta.get("best_loss", best_loss)
    else:
        init = project.ckpt_dir(version.id) / INIT_CKPT
        ref = version.init_ref or {}
        if ref.get("version_id"):
            init = project.ckpt_dir(ref["version_id"]) / ref.get("checkpoint", LAST_CKPT)
        if init.exists():
            try:
                payload = load_checkpoint(init, device)
                model.load_state_dict(payload["model_state"], strict=True)
            except Exception:
                pass

    return Trainer(
        token_path=project.token_file,
        dtype=dtype,
        n_tokens=n_tokens,
        model=model,
        train_cfg=cfg,
        ckpt_last=ckpt_last,
        ckpt_best=project.ckpt_dir(version.id) / BEST_CKPT,
        initial_step=initial_step,
        initial_tokens_seen=tokens_seen,
        initial_history=history,
        best_loss=best_loss,
        optimizer_state=optimizer_state,
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# 4. Persistance du résultat d'un entraînement
# ---------------------------------------------------------------------------
def persist_run(project: Project, version_id: str, summary: dict,
                train_cfg: dict | None = None) -> Version:
    version = project.version(version_id)
    if version is None:
        raise KeyError(version_id)
    version.steps = int(summary.get("steps", 0))
    version.tokens_seen = int(summary.get("tokens_seen", version.tokens_seen))
    version.last_loss = summary.get("loss")
    version.best_loss = summary.get("best_loss")
    version.history = summary.get("history", version.history)
    version.duration += float(summary.get("duration", 0.0))
    version.device = summary.get("device", version.device)
    version.status = "interrompue" if summary.get("interrupted") else "prête"
    if train_cfg:
        version.train = train_cfg
    project.update_version(version)
    return version


def write_log(project: Project, version_id: str, message: str) -> None:
    path = project.log_path(version_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H:%M:%S")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"[{stamp}] {message}\n")


# ---------------------------------------------------------------------------
# 5. Chargement d'un modèle entraîné (discussion)
# ---------------------------------------------------------------------------
def load_version_model(project: Project, version: Version,
                       device_pref: str = "auto", prefer_best: bool = False) -> tuple[GPT, Tokenizer, torch.device, dict]:
    model = GPT(ModelConfig.from_dict(version.model))
    device = pick_device(device_pref)
    candidates = [BEST_CKPT, LAST_CKPT, INIT_CKPT] if prefer_best else [LAST_CKPT, BEST_CKPT, INIT_CKPT]
    chosen = None
    for name in candidates:
        path = project.ckpt_dir(version.id) / name
        if path.exists():
            chosen = path
            break
    if chosen is None:
        ref = version.init_ref or {}
        if ref.get("version_id"):
            path = project.ckpt_dir(ref["version_id"]) / ref.get("checkpoint", LAST_CKPT)
            if path.exists():
                chosen = path
    if chosen is None:
        raise FileNotFoundError("Cette version n'a pas encore de poids : entraînez-la d'abord.")
    payload = load_checkpoint(chosen, device)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    meta = payload.get("meta", {})
    meta["checkpoint"] = chosen.name
    return model, project.tokenizer(), device, meta


def version_summary_row(project: Project, version: Version) -> dict:
    cfg = ModelConfig.from_dict(version.model)
    return {
        "name": version.name,
        "params": human_params(int(version.model.get("params", 0)) or _param_count(cfg)),
        "steps": version.steps,
        "tokens": version.tokens_seen,
        "loss": version.last_loss,
        "best_loss": version.best_loss,
        "status": version.status,
        "created": version.created,
        "parent": version.parent,
        "device": version.device,
    }


def _param_count(cfg: ModelConfig) -> int:
    return count_parameters(cfg)
