"""Point d'entrée de l'application Genesis Studio."""

from __future__ import annotations

import os
import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from . import APP_NAME, APP_VERSION
from .core import paths
from .core.project import ProjectStore, load_settings, save_settings
from .ui import theme
from .ui.main_window import MainWindow


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    """Affiche les erreurs dans une boîte de dialogue au lieu de les avaler."""
    message = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(message, file=sys.stderr)
    try:
        box = QMessageBox()
        box.setWindowTitle("Erreur inattendue")
        box.setIcon(QMessageBox.Critical)
        box.setText("Une erreur inattendue est survenue.")
        box.setDetailedText(message)
        box.exec()
    except Exception:
        pass


def _set_windows_app_id() -> None:
    """Regroupe la fenêtre sous sa propre icône dans la barre des tâches Windows."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("GenesisStudio.App.1")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv)
    sys.excepthook = _excepthook
    _set_windows_app_id()

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("Genesis Studio")
    app.setStyle("Fusion")
    app.setFont(theme.base_font(10))
    app.setPalette(theme.dark_palette())
    app.setStyleSheet(theme.STYLESHEET)
    app.setWindowIcon(theme.app_icon())

    store = ProjectStore()
    window = MainWindow(store)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
