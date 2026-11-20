"""Test de l'interface : construit une capture de chaque écran (plateforme offscreen).

    QT_QPA_PLATFORM=offscreen python tests/test_ui.py [dossier_sortie]

Le script prépare aussi un projet de démonstration (corpus d'exemple, corpus analysé,
une version entraînée en une poignée de pas) pour que les captures aient du contenu.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "captures")
OUT.mkdir(parents=True, exist_ok=True)
# dossier réutilisé d'un lancement à l'autre (évite de réentraîner à chaque test)
TMP = Path(os.environ.get("GENESIS_DEMO_HOME", str(Path(tempfile.gettempdir()) / "genesis-ui-demo")))
TMP.mkdir(parents=True, exist_ok=True)
os.environ["GENESIS_HOME"] = str(TMP / "home")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from genesis.core import corpus, paths, session  # noqa: E402
from genesis.core.project import ProjectStore  # noqa: E402
from genesis.core.trainer import TrainConfig, TrainHooks  # noqa: E402
from genesis.ui import theme  # noqa: E402
from genesis.ui.main_window import MainWindow  # noqa: E402
from genesis.ui.pages.chat_page import ChatPage  # noqa: E402


def build_demo_project() -> "object":
    store = ProjectStore()
    existing = store.list_projects()
    if existing:
        print("projet de démonstration existant réutilisé :", existing[0].name)
        return existing[0]
    project = store.create("Contes & poèmes", "Petit corpus de démonstration", "bpe")
    project.tokenizer_info["options"] = {"vocab_size": 1024}
    project.save()

    # on duplique les exemples pour obtenir un corpus de démonstration réaliste
    bulk = TMP / "corpus_demo.txt"
    text = "\n".join(p.read_text(encoding="utf-8")
                     for p in sorted((paths.assets_dir() / "exemples").glob("*.txt")))
    bulk.write_text("\n".join([text] * 40), encoding="utf-8")
    entries, errors = corpus.import_paths(project.corpus_dir,
                                          [bulk] + sorted((paths.assets_dir() / "exemples").glob("*.txt")))
    assert not errors, errors
    project.add_corpus_entries(entries)

    stats = session.tokenize_project(project)
    print("tokens:", stats["tokens"], "vocab:", stats["vocab_size"])

    cfg = TrainConfig(epochs=8, batch_size=16, seq_len=96, lr=3e-3,
                      eval_every=60, ckpt_every=150, max_steps=528, warmup=20)
    v1, message, _ = session.build_version(project, preset_key="micro", train_cfg=cfg)
    trainer = session.make_trainer(project, v1, TrainHooks(on_log=lambda m: print("  " + m)), cfg)
    summary = trainer.run()
    session.persist_run(project, v1.id, summary, cfg.to_dict())

    cfg2 = TrainConfig(epochs=3, batch_size=16, seq_len=96, lr=1.5e-3,
                       eval_every=60, ckpt_every=150, max_steps=198, warmup=10)
    v2, message2, transferred = session.build_version(
        project, preset_key="micro", train_cfg=cfg2, base_version_id=v1.id
    )
    print(message2)
    trainer2 = session.make_trainer(project, v2, TrainHooks(on_log=lambda m: None), cfg2)
    summary2 = trainer2.run()
    session.persist_run(project, v2.id, summary2, cfg2.to_dict())
    return store.get(project.id)


def shoot(widget, name: str) -> None:
    app = QApplication.instance()
    for _ in range(3):
        app.processEvents()
    widget.repaint()
    pixmap = widget.grab()
    path = OUT / f"{name}.png"
    pixmap.save(str(path))
    print("capture:", path, pixmap.width(), "x", pixmap.height())


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(theme.base_font(10))
    app.setPalette(theme.dark_palette())
    app.setStyleSheet(theme.STYLESHEET)

    project = build_demo_project()
    window = MainWindow(ProjectStore())
    window.resize(1440, 940)
    window.show()

    shoot(window, "01_projets")

    window.open_project(project)
    page = window.project_pages[project.id]
    shoot(window, "02_corpus")

    page.segments.set_index(1)
    shoot(window, "03_versions")

    page.segments.set_index(2)
    shoot(window, "04_entrainement")

    version = project.versions[-1]
    chat_page = ChatPage(project, version)
    chat_page.resize(1300, 900)
    chat_page.show()
    if chat_page.loader is not None:
        chat_page.loader.wait(30000)
    for _ in range(20):
        app.processEvents()
    chat_page.input.setPlainText("Le petit renard traverse la forêt")
    chat_page.send()
    if chat_page.worker is not None:
        chat_page.worker.wait(60000)
    for _ in range(20):
        app.processEvents()
    shoot(chat_page, "05_discussion")

    print("Captures enregistrées dans", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
