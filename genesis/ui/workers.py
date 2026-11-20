"""Tâches longues exécutées hors du fil graphique (import, tokenisation,
entraînement, génération). Chaque tâche émet ses résultats par signaux Qt.
"""

from __future__ import annotations

import time
import threading
import traceback
from pathlib import Path

import torch
from PySide6.QtCore import QThread, Signal

from ..core import chat, corpus, session
from ..core.model import GPT, ModelConfig, pick_device
from ..core.project import Project, Version
from ..core.tokenizers import Tokenizer
from ..core.trainer import TrainConfig, TrainHooks, Trainer


class ImportWorker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object, object)      # entries, erreurs
    failed = Signal(str)

    def __init__(self, corpus_dir: Path, paths: list[Path]):
        super().__init__()
        self.corpus_dir = corpus_dir
        self.paths = paths

    def run(self) -> None:
        try:
            entries, errors = corpus.import_paths(
                self.corpus_dir,
                self.paths,
                progress=lambda done, total, name: self.progress.emit(done, total, name),
            )
            self.done.emit(entries, errors)
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))


class TokenizeWorker(QThread):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, project: Project):
        super().__init__()
        self.project = project
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            result = session.tokenize_project(
                self.project,
                progress=lambda f, m: self.progress.emit(f, m),
                should_stop=lambda: self._stop,
            )
            self.done.emit(result)
        except InterruptedError:
            self.failed.emit("Analyse annulée.")
        except Exception as exc:
            self.failed.emit(str(exc) or traceback.format_exc(limit=2))


class TrainWorker(QThread):
    """Crée (si besoin) une version puis entraîne le modèle."""

    log = Signal(str)
    step = Signal(object)
    version_created = Signal(str, str)      # version_id, message
    finished_run = Signal(object)           # résumé
    failed = Signal(str)

    def __init__(
        self,
        project: Project,
        version: Version,
        train_cfg: TrainConfig,
        resume_exact: bool = False,
    ):
        super().__init__()
        self.project = project
        self.version = version
        self.train_cfg = train_cfg
        self.resume_exact = resume_exact
        self._trainer: Trainer | None = None
        self._last_emit = 0.0

    def stop(self) -> None:
        if self._trainer is not None:
            self._trainer.stop()

    # -- hooks ---------------------------------------------------------
    def _on_log(self, message: str) -> None:
        session.write_log(self.project, self.version.id, message)
        self.log.emit(message)

    def _on_step(self, payload: dict) -> None:
        now = time.time()
        if now - self._last_emit < 0.12 and payload.get("step") != payload.get("total_steps"):
            return
        self._last_emit = now
        self.step.emit(payload)

    def _on_checkpoint(self, meta: dict) -> None:
        # sauvegarde intermédiaire de l'état de la version
        version = self.project.version(self.version.id)
        if version is None:
            return
        version.steps = int(meta.get("step", version.steps))
        version.tokens_seen = int(meta.get("tokens_seen", version.tokens_seen))
        version.last_loss = meta.get("loss", version.last_loss)
        version.best_loss = meta.get("best_loss", version.best_loss)
        version.history = meta.get("history", version.history)
        version.device = meta.get("device", version.device)
        self.project.update_version(version)

    def run(self) -> None:
        try:
            hooks = TrainHooks(
                on_log=self._on_log, on_step=self._on_step, on_checkpoint=self._on_checkpoint
            )
            self._trainer = session.make_trainer(
                self.project, self.version, hooks, self.train_cfg, resume_exact=self.resume_exact
            )
            summary = self._trainer.run()
            session.persist_run(self.project, self.version.id, summary, self.train_cfg.to_dict())
            self.finished_run.emit(summary)
        except Exception:
            self.failed.emit(traceback.format_exc(limit=4))


class ModelLoadWorker(QThread):
    loaded = Signal(object, object, object, object)   # model, tokenizer, device, meta
    failed = Signal(str)

    def __init__(self, project: Project, version: Version, device_pref: str = "auto"):
        super().__init__()
        self.project = project
        self.version = version
        self.device_pref = device_pref

    def run(self) -> None:
        try:
            tokenizer = self.project.tokenizer()
            expected = int(self.version.model.get("vocab_size", 0))
            if expected and tokenizer.vocab_size != expected:
                raise ValueError(
                    f"Ce modèle a été entraîné avec un vocabulaire de {expected} tokens, "
                    f"mais le tokenizer du projet en compte {tokenizer.vocab_size}. "
                    "Ré-analysez le corpus avec les mêmes réglages de tokenizer, "
                    "ou entraînez une nouvelle version."
                )
            model, tokenizer, device, meta = session.load_version_model(
                self.project, self.version, self.device_pref
            )
            self.loaded.emit(model, tokenizer, device, meta)
        except Exception as exc:
            self.failed.emit(str(exc) or traceback.format_exc(limit=2))


class GenerateWorker(QThread):
    token = Signal(str)
    finished_text = Signal(str)
    failed = Signal(str)

    def __init__(self, model: GPT, tokenizer: Tokenizer, prompt: str,
                 params: chat.SamplingParams, stops: list[str], device: torch.device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.params = params
        self.stops = stops
        self.device = device
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            text = chat.stream_generate(
                self.model,
                self.tokenizer,
                self.prompt,
                self.params,
                on_text=lambda piece: self.token.emit(piece),
                should_stop=lambda: self._stop,
                stops=self.stops,
                device=self.device,
            )
            self.finished_text.emit(text)
        except Exception as exc:
            self.failed.emit(str(exc) or traceback.format_exc(limit=2))


class BenchmarkWorker(QThread):
    """Calibre le matériel : lot maximal en mémoire + débit réel (tokens/s)."""

    progress = Signal(float, str)
    done = Signal(object)          # gpu.BenchmarkResult
    failed = Signal(str)

    def __init__(self, model_cfg: dict, device_pref: str = "auto", seq_len: int = 128):
        super().__init__()
        self.model_cfg = model_cfg
        self.device_pref = device_pref
        self.seq_len = seq_len
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            from ..core import gpu

            result = gpu.run_benchmark(
                self.model_cfg,
                device_pref=self.device_pref,
                seq_len=self.seq_len,
                progress=lambda f, m: self.progress.emit(f, m),
                should_stop=lambda: self._stop,
            )
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc) or traceback.format_exc(limit=1))


class CorpusGenerationWorker(QThread):
    """Génère une suite de textes en tâche de fond (aucun corpus n'est téléversé)."""

    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, config, output_dir: Path):
        super().__init__()
        self.config = config
        self.output_dir = Path(output_dir)
        self._cancel = threading.Event()

    def stop(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            from ..core.generators import generate_corpus

            result = generate_corpus(
                self.config,
                self.output_dir,
                progress=lambda f, message: self.progress.emit(f, message),
                should_stop=self._cancel.is_set,
            )
            self.done.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc(limit=3))


class UpdateCheckWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, current_version: str, repository: str):
        super().__init__()
        self.current_version = current_version
        self.repository = repository

    def run(self) -> None:
        try:
            from ..core.updater import check_for_updates

            self.done.emit(check_for_updates(self.current_version, self.repository))
        except Exception as exc:
            self.failed.emit(str(exc) or traceback.format_exc(limit=2))


class UpdateDownloadWorker(QThread):
    progress = Signal(float, str)
    done = Signal(object, object)  # local patch paths, verified manifests
    failed = Signal(str)

    def __init__(self, plan, data_dir: Path):
        super().__init__()
        self.plan = plan
        self.data_dir = Path(data_dir)
        self._cancel = threading.Event()

    def stop(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            from ..core.updater import download_update_chain, verify_patch_archives

            paths = download_update_chain(
                self.plan, self.data_dir,
                progress=lambda f, m: self.progress.emit(f, m),
                should_stop=self._cancel.is_set,
            )
            manifests = verify_patch_archives(paths, self.plan.current_version)
            self.done.emit(paths, manifests)
        except Exception as exc:
            self.failed.emit(str(exc) or traceback.format_exc(limit=2))
