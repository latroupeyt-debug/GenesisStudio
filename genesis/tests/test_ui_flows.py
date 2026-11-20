"""Test fonctionnel de l'interface : pilote les vrais boutons (sans intervention).

    QT_QPA_PLATFORM=offscreen python tests/test_ui_flows.py

Couvre : création de projet, import de corpus, analyse, entraînement depuis l'onglet
Entraînement, version suivante avec héritage des poids, discussion, suppression d'une
version (confirmation simulée).
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TMP = Path(tempfile.mkdtemp(prefix="genesis-flows-"))
os.environ["GENESIS_HOME"] = str(TMP / "home")

from PySide6.QtWidgets import QApplication  # noqa: E402

from genesis.core import paths  # noqa: E402
from genesis.core.project import ProjectStore  # noqa: E402
from genesis.ui import theme  # noqa: E402
from genesis.ui.dialogs import ConfirmDialog  # noqa: E402
from genesis.ui.main_window import MainWindow  # noqa: E402


def pump(app: QApplication, seconds: float = 0.2) -> None:
    deadline = time.time() + seconds
    while time.time() < deadline:
        app.processEvents()
        time.sleep(0.01)


def wait_until(app: QApplication, predicate, timeout: float = 120.0, label: str = "") -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    raise TimeoutError(f"Délai dépassé : {label}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(theme.dark_palette())
    app.setStyleSheet(theme.STYLESHEET)

    # toutes les confirmations sont acceptées (on simule les clics sur « Confirmer »)
    ConfirmDialog.ask = staticmethod(lambda *a, **k: (True, True))  # type: ignore[assignment]

    store = ProjectStore()
    project = store.create("Test flux", "vérification", "bpe")
    project.tokenizer_info["options"] = {"vocab_size": 512}
    project.save()

    window = MainWindow(store)
    window.resize(1360, 900)
    window.show()
    window.open_project(project)
    page = window.project_pages[project.id]
    pump(app)

    # ---- import de corpus via l'onglet Corpus ----
    corpus_tab = page.corpus_tab
    files = sorted((paths.assets_dir() / "exemples").glob("*.txt"))
    assert files, "corpus d'exemple introuvable"
    corpus_tab.import_paths(files)
    wait_until(app, lambda: len(project.corpus_files) == len(files), 60, "import")
    print("import ok :", len(project.corpus_files), "fichiers")

    # ---- analyse du corpus ----
    corpus_tab.analyze()
    wait_until(app, lambda: project.tokens_ready, 300, "analyse du corpus")
    print("tokens :", project.tokens_meta.get("tokens"), "vocab :",
          project.tokenizer_info.get("vocab_size"))

    # ---- entraînement depuis l'onglet Entraînement ----
    page.segments.set_index(2)
    train = page.train_tab
    train.epochs_spin.setValue(12)
    train.batch_spin.setValue(8)
    train.seq_spin.setValue(64)
    train.ckpt_spin.setValue(20)
    train.eval_spin.setValue(20)
    train._update_summary()
    train.start_training()
    wait_until(app, lambda: not train.is_running() and len(project.versions) == 1, 600,
               "entraînement v1")
    v1 = project.versions[0]
    print(f"v1 : {v1.steps} pas · perte {v1.last_loss:.3f} · statut {v1.status}")
    assert v1.is_trained

    # ---- nouvelle version qui hérite des poids (bouton « Continuer ») ----
    page.versions_tab.refresh()
    page._continue_from(v1)
    pump(app)
    assert train._base_version() is not None, "la version de départ n'est pas présélectionnée"
    train.epochs_spin.setValue(4)
    train._update_summary()
    train.start_training()
    wait_until(app, lambda: not train.is_running() and len(project.versions) == 2, 600,
               "entraînement v2")
    v2 = project.versions[-1]
    assert v2.id != v1.id
    assert v2.init_ref.get("version_id") == v1.id, v2.init_ref
    assert v2.tokens_seen > v1.tokens_seen, (v1.tokens_seen, v2.tokens_seen)
    print(f"v2 : {v2.steps} pas · hérite de {v2.init_ref['version_id'][:6]} · "
          f"tokens cumulés {v2.tokens_seen}")

    # ---- discussion ----
    chat = None
    window.open_chat(project, v2)
    chat = window.stack.currentWidget()
    assert chat.__class__.__name__ == "ChatPage"
    if chat.loader is not None:
        wait_until(app, lambda: chat.model is not None, 120, "chargement du modèle")
    chat.input.setPlainText("Le petit renard")
    chat.send()
    wait_until(app, lambda: chat.worker is not None and chat.worker.isFinished(), 180,
               "génération")
    pump(app, 0.3)
    assert chat.turns and chat.turns[-1][0] == "assistant", chat.turns
    print("réponse :", repr(chat.turns[-1][1][:60]))
    window._close_chat(project)
    pump(app)

    # ---- suppression d'une version (confirmation simulée) ----
    page.segments.set_index(1)
    page.versions_tab.refresh()
    page.versions_tab.delete_version(v1)
    pump(app, 0.3)
    assert len(project.versions) == 1, [v.name for v in project.versions]
    assert not project.version_dir(v1.id).exists()
    print("suppression ok :", [v.name for v in project.versions])

    # ---- réouverture du projet : l'état est bien persisté ----
    store2 = ProjectStore()
    reloaded = store2.get(project.id)
    assert reloaded is not None and len(reloaded.versions) == 1
    assert reloaded.tokens_ready
    print("persistance ok")

    window.close()
    pump(app, 0.2)
    print("\nTOUS LES TESTS D'INTERFACE SONT PASSÉS")
    print("Dossier :", TMP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
