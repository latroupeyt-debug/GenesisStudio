"""Captures complémentaires : état vide, boîte de création, confirmation de suppression,
détails d'une version, aide. Nécessite un dossier de démonstration existant
(tests/test_ui.py le crée).

    QT_QPA_PLATFORM=offscreen python tests/test_dialogs.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "captures"
OUT.mkdir(parents=True, exist_ok=True)

# 1) état vide (aucun projet) ------------------------------------------------
EMPTY_HOME = Path(tempfile.mkdtemp(prefix="genesis-empty-"))
os.environ["GENESIS_HOME"] = str(EMPTY_HOME / "home")

from PySide6.QtWidgets import QApplication  # noqa: E402

from genesis.core.project import ProjectStore  # noqa: E402
from genesis.ui import theme  # noqa: E402
from genesis.ui.dialogs import (  # noqa: E402
    AboutDialog,
    ConfirmDialog,
    CorpusGeneratorDialog,
    NewProjectDialog,
    SettingsDialog,
    SizingGuideDialog,
    VersionDetailsDialog,
)
from genesis.ui.main_window import MainWindow  # noqa: E402


def shoot(widget, name: str) -> None:
    app = QApplication.instance()
    for _ in range(4):
        app.processEvents()
    widget.repaint()
    path = OUT / f"{name}.png"
    widget.grab().save(str(path))
    print("capture:", path)


class _AutoClose:
    """Referme une boîte modale dès qu'elle est affichée, après la capture."""

    def __init__(self, name: str):
        self.name = name

    def __call__(self, dialog) -> None:
        pass


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(theme.dark_palette())
    app.setStyleSheet(theme.STYLESHEET)

    window = MainWindow(ProjectStore())
    window.resize(1440, 940)
    window.show()
    shoot(window, "06_etat_vide")

    dialog = NewProjectDialog(window)
    dialog.name_input.setText("Code Python")
    dialog.desc_input.setPlainText("Un modèle entraîné sur des sources Python.")
    dialog.select("bpe")
    dialog.resize(640, 900)
    dialog.show()
    shoot(dialog, "07_nouveau_projet")
    dialog.close()

    confirm = ConfirmDialog(
        window, "Supprimer la version v3 ?",
        "Cette version et tous ses checkpoints seront définitivement supprimés. "
        "Les autres versions du projet ne sont pas affectées.",
        confirm_text="Supprimer la version",
        detail="2 fichier(s) de poids · 22 Mo · 1 250 pas d'entraînement",
        checkbox="Je confirme la suppression définitive de cette version",
    )
    confirm.resize(520, 300)
    confirm.show()
    shoot(confirm, "08_confirmation_suppression")
    confirm.close()

    about = AboutDialog(window)
    about.show()
    shoot(about, "09_a_propos")
    about.close()

    settings = SettingsDialog(window, {"device_index": 0})
    settings.resize(680, 760)
    settings.show()
    shoot(settings, "11_reglages_gpu")
    settings.close()

    generator_project = ProjectStore().create("Projet générateur", "")
    generator = CorpusGeneratorDialog(window, generator_project)
    generator.resize(700, 820)
    generator.show()
    shoot(generator, "13_generateur_corpus")
    generator.close()

    from genesis.core import sizing

    guide = SizingGuideDialog(window, 2048, 4.0, sizing.advise(951_424, 103_492, 2))
    guide.resize(760, 720)
    guide.show()
    shoot(guide, "12_guide_tailles")
    guide.close()

    # 2) pages nécessitant un projet de démonstration ------------------------
    demo_home = Path(tempfile.gettempdir()) / "genesis-ui-demo" / "home"
    if (demo_home / "projects").exists():
        os.environ["GENESIS_HOME"] = str(demo_home)
        from genesis.core import paths as paths_mod

        store = ProjectStore(str(demo_home / "projects"))
        projects = store.list_projects()
        if projects:
            project = projects[0]
            if project.versions:
                details = VersionDetailsDialog(window, project, project.versions[-1])
                details.show()
                shoot(details, "10_details_version")
                details.close()
    print("Captures complémentaires terminées.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
