"""Chemins et arborescence de travail de l'application.

Toutes les données utilisateur (projets, corpus, versions, checkpoints) vivent
dans un seul dossier racine, facile à sauvegarder ou à déplacer.

Emplacement par défaut :
    Windows : %LOCALAPPDATA%\\GenesisStudio
    macOS   : ~/Library/Application Support/GenesisStudio
    Linux   : ~/.local/share/GenesisStudio

Priorité :
    1. variable d'environnement GENESIS_HOME
    2. mode portable (fichier `portable.txt` à côté de l'exécutable)
    3. dossier système ci-dessus
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_FOLDER = "GenesisStudio"


def _portable_root() -> Path | None:
    """Mode portable : données à côté de l'exécutable."""
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[2]
    flag = base / "portable.txt"
    if flag.exists():
        return base / "data"
    return None


def home() -> Path:
    env = os.environ.get("GENESIS_HOME")
    if env:
        root = Path(env).expanduser().resolve()
    else:
        portable = _portable_root()
        if portable is not None:
            root = portable
        elif sys.platform == "win32":
            root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APP_FOLDER
        elif sys.platform == "darwin":
            root = Path.home() / "Library" / "Application Support" / APP_FOLDER
        else:
            root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_FOLDER
    root.mkdir(parents=True, exist_ok=True)
    return root


def projects_dir() -> Path:
    p = home() / "projects"
    p.mkdir(parents=True, exist_ok=True)
    return p


def settings_file() -> Path:
    return home() / "settings.json"


def trash_dir() -> Path:
    p = home() / "corbeille"
    p.mkdir(parents=True, exist_ok=True)
    return p


def assets_dir() -> Path:
    """Dossier des ressources embarquées (exemples de corpus, icônes)."""
    base = getattr(sys, "_MEIPASS", None)
    if base:
        return Path(base) / "assets"
    return Path(__file__).resolve().parents[2] / "assets"


def free_space_gb(path: Path | None = None) -> float:
    import shutil

    try:
        return shutil.disk_usage(str(path or home())).free / (1024 ** 3)
    except OSError:
        return 0.0


def human_size(num_bytes: float) -> str:
    step = 1024.0
    for unit in ("o", "Ko", "Mo", "Go", "To"):
        if abs(num_bytes) < step or unit == "To":
            if unit == "o":
                return f"{int(num_bytes)} o"
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= step
    return f"{num_bytes:.1f} To"


def human_count(n: float) -> str:
    for limit, suffix in ((1e9, "Md"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= limit:
            return f"{n / limit:.1f}{suffix}".replace(".0", "")
    return f"{int(n)}"


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} s"
    if seconds < 3600:
        return f"{seconds // 60} min {seconds % 60:02d} s"
    return f"{seconds // 3600} h {(seconds % 3600) // 60:02d}"
