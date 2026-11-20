"""Modèle de données : projets, versions, corpus, réglages.

Un projet = un tokenizer + un corpus + une lignée de versions du modèle.
Chaque version garde son propre dossier (configuration, checkpoints, journal),
ce qui permet de reprendre l'entraînement depuis n'importe quelle version.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from . import paths
from .tokenizers import Tokenizer, create_tokenizer

PROJECT_FILE = "project.json"
FORMAT_VERSION = 1


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def pretty_date(iso: str | None) -> str:
    if not iso:
        return "—"
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError:
        return iso
    return dt.strftime("%d/%m/%Y à %H:%M")


def slugify(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip().lower()
    text = re.sub(r"[\s_-]+", "-", text)
    return text[:40] or "projet"


def _atomic_write(path: Path, data: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Version du modèle
# ---------------------------------------------------------------------------
@dataclass
class Version:
    id: str
    name: str
    created: str
    parent: str | None = None
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    status: str = "jamais entraînée"   # jamais entraînée | interrompue | prête
    steps: int = 0
    tokens_seen: int = 0
    best_loss: float | None = None
    last_loss: float | None = None
    duration: float = 0.0
    device: str = "cpu"
    note: str = ""
    init_ref: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # [[step, loss, eval_loss|null], ...]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "created": self.created,
            "parent": self.parent, "model": self.model, "train": self.train,
            "status": self.status, "steps": self.steps,
            "tokens_seen": self.tokens_seen, "best_loss": self.best_loss,
            "last_loss": self.last_loss, "duration": self.duration,
            "device": self.device, "note": self.note, "init_ref": self.init_ref,
            "history": self.history,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Version":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def params_label(self) -> str:
        return self.model.get("preset_label", "—")

    @property
    def is_trained(self) -> bool:
        return self.steps > 0


# ---------------------------------------------------------------------------
# Projet
# ---------------------------------------------------------------------------
class Project:
    def __init__(self, folder: Path, data: dict):
        self.folder = Path(folder)
        self.data = data
        self._tokenizer: Tokenizer | None = None

    # -- accès ---------------------------------------------------------
    @property
    def id(self) -> str:
        return self.data["id"]

    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def description(self) -> str:
        return self.data.get("description", "")

    @property
    def created(self) -> str:
        return self.data.get("created", now_iso())

    @property
    def updated(self) -> str:
        return self.data.get("updated", self.created)

    @property
    def corpus_dir(self) -> Path:
        return self.folder / "corpus"

    @property
    def versions_dir(self) -> Path:
        return self.folder / "versions"

    @property
    def tokenizer_dir(self) -> Path:
        return self.folder / "tokenizer"

    @property
    def tokenizer_info(self) -> dict:
        return self.data.setdefault("tokenizer", {"kind": "char", "trained": False})

    @property
    def tokenizer_kind(self) -> str:
        return self.tokenizer_info.get("kind", "char")

    @property
    def corpus_files(self) -> list[dict]:
        return self.data.setdefault("corpus", [])

    @property
    def tokens_meta(self) -> dict:
        return self.data.setdefault("tokens", {})

    @property
    def token_file(self) -> Path:
        return self.folder / "tokens.bin"

    @property
    def versions(self) -> list[Version]:
        return [Version.from_dict(v) for v in self.data.get("versions", [])]

    def version(self, version_id: str) -> Version | None:
        for v in self.versions:
            if v.id == version_id:
                return v
        return None

    def version_raw(self, version_id: str) -> dict | None:
        for v in self.data.get("versions", []):
            if v["id"] == version_id:
                return v
        return None

    def version_dir(self, version_id: str) -> Path:
        return self.versions_dir / version_id

    def ckpt_dir(self, version_id: str) -> Path:
        return self.version_dir(version_id) / "checkpoints"

    def log_path(self, version_id: str) -> Path:
        return self.version_dir(version_id) / "entrainement.log"

    # -- tokenizer ------------------------------------------------------
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            if (self.tokenizer_dir / "tokenizer.json").exists():
                self._tokenizer = Tokenizer.load(self.tokenizer_dir)
            else:
                self._tokenizer = create_tokenizer(self.tokenizer_kind)
        return self._tokenizer

    def set_tokenizer(self, tokenizer: Tokenizer) -> None:
        tokenizer.save(self.tokenizer_dir)
        self._tokenizer = tokenizer
        self.tokenizer_info.update(
            {"kind": tokenizer.kind, "trained": True, "vocab_size": tokenizer.vocab_size}
        )
        self.save()

    # -- corpus ---------------------------------------------------------
    def corpus_signature(self) -> str:
        parts = [f"{f['stored']}:{f.get('size', 0)}" for f in self.corpus_files]
        parts.append(f"tok={self.tokenizer_kind}:{self.tokenizer_info.get('vocab_size', 0)}")
        return "|".join(sorted(parts))

    @property
    def tokens_ready(self) -> bool:
        meta = self.tokens_meta
        return bool(meta.get("tokens")) and self.token_file.exists() and meta.get("signature") == self.corpus_signature()

    def add_corpus_entries(self, entries: Iterable[dict]) -> None:
        self.corpus_files.extend(entries)
        self.invalidate_tokens()
        self.save()

    def remove_corpus_file(self, stored: str) -> None:
        self.data["corpus"] = [f for f in self.corpus_files if f["stored"] != stored]
        path = self.corpus_dir / stored
        if path.exists():
            path.unlink()
        self.invalidate_tokens()
        self.save()

    def invalidate_tokens(self) -> None:
        self.data["tokens"] = {}
        if self.token_file.exists():
            try:
                self.token_file.unlink()
            except OSError:
                pass
        self.save()

    # -- versions -------------------------------------------------------
    def next_version_name(self) -> str:
        existing = {v.name for v in self.versions}
        i = len(existing) + 1
        while f"v{i}" in existing:
            i += 1
        return f"v{i}"

    def create_version(self, model_cfg: dict, train_cfg: dict,
                       parent: str | None = None, name: str | None = None) -> Version:
        version_id = "v_" + uuid.uuid4().hex[:10]
        version = Version(
            id=version_id,
            name=name or self.next_version_name(),
            created=now_iso(),
            parent=parent,
            model=model_cfg,
            train=train_cfg,
        )
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.version_dir(version_id).mkdir(parents=True, exist_ok=True)
        self.ckpt_dir(version_id).mkdir(parents=True, exist_ok=True)
        self.data.setdefault("versions", []).append(version.to_dict())
        self.save()
        return version

    def update_version(self, version: Version) -> None:
        raw = self.version_raw(version.id)
        if raw is not None:
            raw.update(version.to_dict())
        self.touch()
        self.save()

    def delete_version(self, version_id: str) -> None:
        self.data["versions"] = [v for v in self.data.get("versions", []) if v["id"] != version_id]
        folder = self.version_dir(version_id)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
        self.save()

    # -- divers ---------------------------------------------------------
    def touch(self) -> None:
        self.data["updated"] = now_iso()

    def save(self) -> None:
        self.data["updated"] = now_iso()
        _atomic_write(self.folder / PROJECT_FILE, json.dumps(self.data, ensure_ascii=False, indent=2))

    def rename(self, name: str, description: str | None = None) -> None:
        self.data["name"] = name.strip() or self.name
        if description is not None:
            self.data["description"] = description.strip()
        self.save()

    def open_in_explorer(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder)))

    def summary(self) -> dict:
        from .corpus import corpus_stats

        stats = corpus_stats(self.corpus_files)
        stats["versions"] = len(self.versions)
        stats["tokens"] = self.tokens_meta.get("tokens", 0)
        stats["tokens_ready"] = self.tokens_ready
        stats["vocab"] = self.tokenizer_info.get("vocab_size", 0)
        stats["tokenizer_trained"] = bool(self.tokenizer_info.get("trained"))
        return stats


# ---------------------------------------------------------------------------
# Gestion des projets sur disque
# ---------------------------------------------------------------------------
class ProjectStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else paths.projects_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, Project] = {}

    def list_projects(self) -> list[Project]:
        projects: list[Project] = []
        for folder in sorted(self.root.iterdir()):
            if not folder.is_dir():
                continue
            project_file = folder / PROJECT_FILE
            if not project_file.exists():
                continue
            try:
                data = json.loads(project_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            projects.append(Project(folder, data))
        projects.sort(key=lambda p: p.updated, reverse=True)
        return projects

    def get(self, project_id: str) -> Project | None:
        for project in self.list_projects():
            if project.id == project_id:
                return project
        return None

    def create(self, name: str, description: str = "", tokenizer_kind: str = "char") -> Project:
        project_id = "p_" + uuid.uuid4().hex[:10]
        folder = self.root / f"{slugify(name)}-{project_id[-4:]}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "corpus").mkdir(exist_ok=True)
        (folder / "versions").mkdir(exist_ok=True)
        (folder / "tokenizer").mkdir(exist_ok=True)
        data = {
            "format": FORMAT_VERSION,
            "id": project_id,
            "name": name.strip() or "Nouveau projet",
            "description": description.strip(),
            "created": now_iso(),
            "updated": now_iso(),
            "tokenizer": {"kind": tokenizer_kind, "trained": False},
            "corpus": [],
            "tokens": {},
            "versions": [],
        }
        project = Project(folder, data)
        project.save()
        return project

    def delete(self, project: Project, hard: bool = False) -> None:
        if hard:
            shutil.rmtree(project.folder, ignore_errors=True)
            return
        # suppression douce : déplacement dans la corbeille de l'application
        target = paths.trash_dir() / f"{project.folder.name}-{int(time.time())}"
        try:
            shutil.move(str(project.folder), str(target))
        except OSError:
            shutil.rmtree(project.folder, ignore_errors=True)

    def duplicate(self, project: Project) -> Project:
        copy = self.create(project.name + " (copie)", project.description, project.tokenizer_kind)
        shutil.rmtree(copy.folder)
        shutil.copytree(project.folder, copy.folder)
        data = json.loads((copy.folder / PROJECT_FILE).read_text(encoding="utf-8"))
        data["id"] = copy.id
        data["created"] = now_iso()
        data["updated"] = now_iso()
        copy.data = data
        copy.save()
        return copy


# ---------------------------------------------------------------------------
# Réglages applicatifs
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "last_project": "",
    "threads": 0,            # 0 = automatique
    "device": "auto",        # auto | cpu | cuda
    "max_log_lines": 400,
    "show_tips": True,
}


def load_settings() -> dict:
    path = paths.settings_file()
    data = dict(DEFAULT_SETTINGS)
    if path.exists():
        try:
            data.update(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return data


def save_settings(data: dict) -> None:
    _atomic_write(paths.settings_file(), json.dumps(data, ensure_ascii=False, indent=2))
