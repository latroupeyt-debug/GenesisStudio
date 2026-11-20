"""Smoke test offscreen du panneau de génération : aucun appel réseau réel."""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication  # noqa: E402

from genesis.core.project import ProjectStore  # noqa: E402
from genesis.ui.dialogs import CorpusGeneratorDialog  # noqa: E402
from genesis.ui.pages.corpus_tab import CorpusTab  # noqa: E402
from genesis.ui.theme import dark_palette  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="genesis-generator-ui-") as tmp:
        os.environ["GENESIS_HOME"] = tmp
        app = QApplication.instance() or QApplication([])
        app.setStyle("Fusion")
        app.setPalette(dark_palette())
        project = ProjectStore().create("Corpus test", "")
        dialog = CorpusGeneratorDialog(None, project)
        assert dialog.provider_combo.currentData() == "ollama"
        assert dialog.key_edit.isHidden()
        dialog.provider_combo.setCurrentIndex(1)
        assert dialog.provider_combo.currentData() == "groq"
        assert not dialog.remote_consent.isHidden()
        dialog.start_generation()
        assert "Confirmez" in dialog.status.text()
        dialog.remote_consent.setChecked(True)
        dialog.start_generation()
        assert "clé API" in dialog.status.text()
        dialog.provider_combo.setCurrentIndex(2)
        dialog.key_edit.setText("test-key")
        dialog.remote_consent.setChecked(True)
        dialog.model_edit.setText("model-that-might-cost")
        dialog.start_generation()
        assert "finissant par :free" in dialog.status.text()
        dialog.close()

        # Parcours de bout en bout avec une sortie déjà générée (aucun réseau/API).
        generated_dir = project.folder / ".generated"
        generated_dir.mkdir(parents=True, exist_ok=True)
        generated_file = generated_dir / "generation_ia_test.txt"
        generated_file.write_text(
            "Un texte de démonstration suffisamment long pour être importé dans le corpus. "
            "Il contient plusieurs phrases différentes et peut être relu avant l'entraînement.",
            encoding="utf-8",
        )
        result = SimpleNamespace(
            path=generated_file, error="", stopped=False, accepted=1, requests=1,
            approx_tokens=100, target_tokens=100, summary=lambda: "terminé · 1 texte",
        )

        class FakeGeneratorDialog:
            def __init__(self, *args, **kwargs):
                self.result = result
            def exec(self):
                return 1

        tab = CorpusTab(project)
        with patch("genesis.ui.pages.corpus_tab.CorpusGeneratorDialog", FakeGeneratorDialog):
            tab.open_generator()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            app.processEvents()
            worker = tab._worker
            if worker is not None and not worker.isRunning():
                app.processEvents()
                break
            time.sleep(0.02)
        assert len(project.corpus_files) == 1, project.corpus_files
        assert not generated_file.exists()
        assert (project.corpus_dir / project.corpus_files[0]["stored"]).exists()
        print("UI générateur : consentement, garde-fous et import du fichier final OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
