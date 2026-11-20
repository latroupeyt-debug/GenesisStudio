"""Onglet « Corpus » : import par glisser-déposer, conversion, analyse en tokens."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core import corpus, paths, session
from ...core.project import Project
from .. import theme
from ..dialogs import ConfirmDialog, CorpusGeneratorDialog, TokenizerDialog
from ..widgets import (
    Card,
    Chip,
    Collapsible,
    DropZone,
    IconButton,
    ScrollArea,
    StatTile,
    dim,
    h2,
    h3,
    label,
    muted,
    separator,
    toast,
)
from ..workers import ImportWorker, TokenizeWorker

FORMATS_HINT = ("Formats acceptés : texte (.txt), Markdown (.md), CSV/TSV, JSON/JSONL, HTML, "
                "XML, EPUB, LaTeX — conversion automatique en texte brut. "
                "(.docx et .pdf si python-docx / pypdf sont installés.)")


class FileRow(QFrame):
    removed = Signal(str)
    preview = Signal(str)

    def __init__(self, info: dict, parent: QWidget | None = None):
        super().__init__(parent)
        self.info = info
        self.setObjectName("CardSoft")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 9, 10, 9)
        layout.setSpacing(12)

        icon = QLabel()
        icon.setPixmap(theme.svg_icon("file", theme.TEXT_MUTED, 17).pixmap(17, 17))
        layout.addWidget(icon)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        name = QLabel(info["name"])
        name.setStyleSheet("font-weight: 600;")
        titles.addWidget(name)
        detail = QLabel(
            f"{info.get('format', 'Fichier')} · {paths.human_size(info.get('size', 0))} · "
            f"{info.get('chars', 0):,} caractères".replace(",", " ")
        )
        detail.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        titles.addWidget(detail)
        layout.addLayout(titles, 1)

        preview_btn = IconButton("info", "Aperçu du texte converti")
        preview_btn.clicked.connect(lambda: self.preview.emit(info["stored"]))
        layout.addWidget(preview_btn)

        remove_btn = IconButton("trash", "Retirer du corpus")
        remove_btn.clicked.connect(lambda: self.removed.emit(info["stored"]))
        layout.addWidget(remove_btn)


class CorpusTab(QWidget):
    changed = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self._worker: object | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        scroll = ScrollArea()
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 4, 8, 24)
        layout.setSpacing(16)

        # -- statistiques ------------------------------------------------
        stats_row = QHBoxLayout()
        stats_row.setSpacing(12)
        self.tile_files = StatTile("0", "fichiers")
        self.tile_size = StatTile("0 o", "volume sur disque")
        self.tile_chars = StatTile("0", "caractères")
        self.tile_tokens = StatTile("—", "tokens encodés", theme.MINT)
        for tile in (self.tile_files, self.tile_size, self.tile_chars, self.tile_tokens):
            stats_row.addWidget(tile, 1)
        layout.addLayout(stats_row)

        # -- import ------------------------------------------------------
        import_card = Card(margins=18, spacing=12)
        self.drop = DropZone(
            "Glissez vos fichiers ici",
            "Fichiers texte, dossiers entiers… ou cliquez pour parcourir. "
            "Les formats structurés sont convertis automatiquement en texte.",
        )
        self.drop.filesDropped.connect(self.import_paths)
        self.drop.clicked.connect(self.browse_files)
        import_card.body.addWidget(self.drop)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        browse = QPushButton("  Parcourir les fichiers")
        browse.setIcon(theme.svg_icon("folder", theme.TEXT, 16))
        browse.setCursor(Qt.PointingHandCursor)
        browse.clicked.connect(self.browse_files)
        folder = QPushButton("  Ajouter un dossier")
        folder.setIcon(theme.svg_icon("layers", theme.TEXT, 16))
        folder.setCursor(Qt.PointingHandCursor)
        folder.clicked.connect(self.browse_folder)
        buttons.addWidget(browse)
        buttons.addWidget(folder)
        buttons.addStretch(1)
        example = QPushButton("  Corpus d'exemple")
        example.setObjectName("Ghost")
        example.setIcon(theme.svg_icon("spark", theme.ACCENT, 15))
        example.setToolTip("Ajouter deux petits corpus de démonstration (poèmes, contes)")
        example.setCursor(Qt.PointingHandCursor)
        example.clicked.connect(self.import_examples)
        buttons.addWidget(example)
        generate = QPushButton("  Générer avec une IA")
        generate.setObjectName("Ghost")
        generate.setIcon(theme.svg_icon("spark", theme.ACCENT, 15))
        generate.setToolTip("Créer des textes originaux avec Ollama local, Groq ou OpenRouter")
        generate.setCursor(Qt.PointingHandCursor)
        generate.clicked.connect(self.open_generator)
        buttons.addWidget(generate)
        import_card.body.addLayout(buttons)
        import_card.body.addWidget(dim(FORMATS_HINT))
        layout.addWidget(import_card)

        # -- liste des fichiers ------------------------------------------
        files_card = Card(margins=18, spacing=10)
        files_head = QHBoxLayout()
        files_head.addWidget(h3("Fichiers du corpus"))
        self.files_count = Chip("0", "neutral")
        files_head.addWidget(self.files_count)
        files_head.addStretch(1)
        clear_btn = QPushButton("Tout retirer")
        clear_btn.setObjectName("Ghost")
        clear_btn.clicked.connect(self.clear_files)
        files_head.addWidget(clear_btn)
        files_card.body.addLayout(files_head)
        self.files_container = QWidget()
        self.files_layout = QVBoxLayout(self.files_container)
        self.files_layout.setContentsMargins(0, 0, 0, 0)
        self.files_layout.setSpacing(8)
        files_card.body.addWidget(self.files_container)
        self.files_empty = dim("Aucun fichier pour l'instant. Glissez vos textes ci-dessus pour "
                               "commencer.")
        files_card.body.addWidget(self.files_empty)
        self.import_progress = QProgressBar()
        self.import_progress.setRange(0, 0)
        self.import_progress.setVisible(False)
        self.import_progress.setFixedHeight(6)
        files_card.body.addWidget(self.import_progress)
        self.import_status = dim("")
        self.import_status.setVisible(False)
        files_card.body.addWidget(self.import_status)
        layout.addWidget(files_card)

        # -- analyse -----------------------------------------------------
        analysis_card = Card(margins=18, spacing=12)
        head = QHBoxLayout()
        head.addWidget(h3("Analyse du corpus"))
        self.state_chip = Chip("non analysé", "amber")
        head.addWidget(self.state_chip)
        head.addStretch(1)
        self.analyze_btn = QPushButton("  Analyser le corpus")
        self.analyze_btn.setObjectName("Primary")
        self.analyze_btn.setIcon(theme.svg_icon("bolt", "#FFFFFF", 16))
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.clicked.connect(self.analyze)
        head.addWidget(self.analyze_btn)
        analysis_card.body.addLayout(head)

        self.analysis_help = muted(
            "L'analyse entraîne le tokenizer sur votre corpus puis convertit tous les textes "
            "en un fichier de tokens. C'est cette étape qui permet à l'entraînement de lire le "
            "corpus par petits blocs, sans jamais saturer la mémoire. "
            "Comptez de quelques secondes à deux minutes selon la taille du corpus "
            "(l'apprentissage BPE est le plus long)."
        )
        analysis_card.body.addWidget(self.analysis_help)

        self.analysis_progress = QProgressBar()
        self.analysis_progress.setRange(0, 100)
        self.analysis_progress.setValue(0)
        self.analysis_progress.setVisible(False)
        self.analysis_progress.setFixedHeight(8)
        analysis_card.body.addWidget(self.analysis_progress)
        self.analysis_status = dim("")
        self.analysis_status.setVisible(False)
        analysis_card.body.addWidget(self.analysis_status)

        analysis_card.body.addWidget(separator())
        tok_row = QHBoxLayout()
        tok_row.setSpacing(10)
        self.tokenizer_label = label("", "Muted")
        tok_row.addWidget(self.tokenizer_label, 1)
        tok_btn = QPushButton("Modifier le tokenizer")
        tok_btn.setObjectName("Ghost")
        tok_btn.setCursor(Qt.PointingHandCursor)
        tok_btn.clicked.connect(self.edit_tokenizer)
        tok_row.addWidget(tok_btn)
        analysis_card.body.addLayout(tok_row)
        self.tokenizer_hint = dim("")
        analysis_card.body.addWidget(self.tokenizer_hint)
        layout.addWidget(analysis_card)

        # -- aperçu ------------------------------------------------------
        preview_body = QWidget()
        preview_layout = QVBoxLayout(preview_body)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_label = QLabel("")
        self.preview_label.setWordWrap(True)
        self.preview_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.preview_label.setStyleSheet(
            f"background: #0C1016; border: 1px solid {theme.BORDER_SOFT}; border-radius: 10px;"
            f" padding: 12px; color: {theme.TEXT_MUTED}; font-family: 'Cascadia Mono', 'Consolas',"
            " 'DejaVu Sans Mono', monospace; font-size: 11px;"
        )
        self.preview_label.setMaximumHeight(220)
        preview_layout.addWidget(self.preview_label)
        layout.addWidget(Collapsible("Aperçu du texte converti", preview_body))

        layout.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll)

        self.refresh()

    # -- affichage ------------------------------------------------------
    def refresh(self) -> None:
        files = self.project.corpus_files
        stats = self.project.summary()

        while self.files_layout.count():
            item = self.files_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        for info in files:
            row = FileRow(info)
            row.removed.connect(self.remove_file)
            row.preview.connect(self.preview_file)
            self.files_layout.addWidget(row)

        self.files_empty.setVisible(not files)
        self.files_count.setText(str(len(files)))
        self.tile_files.set_value(str(stats["files"]))
        self.tile_size.set_value(paths.human_size(stats["bytes"]))
        self.tile_chars.set_value(paths.human_count(stats["chars"]))
        self.tile_tokens.set_value(
            paths.human_count(stats["tokens"]) if stats["tokens_ready"] else "—"
        )

        info = self.project.tokenizer_info
        names = {"char": "Caractères", "word": "Mots", "bpe": "BPE (sous-mots)"}
        options = info.get("options", {})
        extra = ""
        if info.get("kind") == "word" and options.get("max_vocab"):
            extra = f" · {options['max_vocab']:,} mots max".replace(",", " ")
        elif info.get("kind") == "bpe" and options.get("vocab_size"):
            extra = f" · {options['vocab_size']:,} tokens visés".replace(",", " ")
        self.tokenizer_label.setText(
            f"Tokenizer : {names.get(self.project.tokenizer_kind, self.project.tokenizer_kind)}{extra}"
        )
        self.tokenizer_hint.setText(
            f"Vocabulaire actuel : {stats['vocab']:,} entrées.".replace(",", " ")
            if stats["vocab"] else "Le vocabulaire sera calculé lors de l'analyse."
        )

        ready = stats["tokens_ready"]
        stale = bool(files) and not ready and bool(stats["tokens"])
        if ready:
            self.state_chip.setText(f"prêt · {paths.human_count(stats['tokens'])} tokens")
            self.state_chip.set_tone("mint")
            self.analyze_btn.setText("  Ré-analyser le corpus")
            self.analysis_help.setText(
                f"Corpus analysé le {self.project.tokens_meta.get('built_at', '—')} : "
                f"{stats['tokens']:,} tokens, vocabulaire de {stats['vocab']:,} entrées. "
                "Vous pouvez entraîner un modèle.".replace(",", " ")
            )
        elif stale:
            self.state_chip.setText("à ré-analyser", "amber")
            self.state_chip.set_tone("amber")
            self.analyze_btn.setText("  Analyser le corpus")
        else:
            self.state_chip.setText("non analysé")
            self.state_chip.set_tone("amber")
            self.analyze_btn.setText("  Analyser le corpus")

        self.analyze_btn.setEnabled(bool(files))
        self.clear_visible = bool(files)

        if self.project.corpus_files:
            self.preview_label.setText(corpus.preview_corpus(files, self.project.corpus_dir) or "—")
        else:
            self.preview_label.setText("—")

    # -- import ---------------------------------------------------------
    def browse_files(self) -> None:
        filters = "Textes et documents (" + " ".join(f"*{e}" for e in sorted(corpus.SUPPORTED_EXTS)) + ");;Tous les fichiers (*)"
        files, _ = QFileDialog.getOpenFileNames(self, "Ajouter des fichiers au corpus", "", filters)
        if files:
            self.import_paths([Path(f) for f in files])

    def browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Ajouter un dossier entier")
        if folder:
            self.import_paths([Path(folder)])

    def open_generator(self) -> None:
        dialog = CorpusGeneratorDialog(self.window(), self.project)
        if dialog.exec() and dialog.result and dialog.result.path:
            result = dialog.result
            self._generated_notice = (
                f"Génération IA : {result.summary()}"
                + (f" — {result.error}" if result.error else "")
            )
            self._generated_tone = (
                "amber" if result.error or result.stopped or result.approx_tokens < result.target_tokens
                else "mint"
            )
            self.import_paths([result.path])

    def import_examples(self) -> None:
        folder = paths.assets_dir() / "exemples"
        files = sorted(folder.glob("*.txt")) if folder.exists() else []
        if not files:
            toast(self.window(), "Corpus d'exemple introuvable.", "danger")
            return
        self.import_paths(files)

    def import_paths(self, paths_list: list[Path]) -> None:
        valid = [p for p in paths_list if corpus.is_supported(p)]
        skipped = [p for p in paths_list if not corpus.is_supported(p)]
        if not valid:
            toast(self.window(), "Aucun fichier compatible dans cette sélection.", "danger")
            return

        self.import_progress.setVisible(True)
        self.import_status.setVisible(True)
        self.import_status.setText("Import en cours…")
        worker = ImportWorker(self.project.corpus_dir, valid)
        self._worker = worker

        def on_progress(done: int, total: int, name: str) -> None:
            self.import_status.setText(f"Import {done + 1}/{total} · {name}")

        def on_done(entries: list, errors: list) -> None:
            self.import_progress.setVisible(False)
            self.import_status.setVisible(False)
            if entries:
                self.project.add_corpus_entries(entries)
                self.refresh()
                self.changed.emit()
                toast(self.window(),
                      f"{len(entries)} fichier(s) ajouté(s) — pensez à analyser le corpus.",
                      "mint")
                notice = getattr(self, "_generated_notice", "")
                if notice:
                    toast(self.window(), notice, getattr(self, "_generated_tone", "mint"))
            self._generated_notice = ""
            if errors:
                ConfirmDialog.ask(
                    self.window(), "Certains fichiers n'ont pas pu être lus",
                    "\n".join(errors[:8]), confirm_text="Compris", danger=False,
                    detail="Les autres fichiers ont bien été ajoutés.",
                )
            # Les sorties temporaires de l'assistant sont copiées dans le corpus par import_file.
            for source in valid:
                if source.parent == self.project.folder / ".generated":
                    try:
                        source.unlink(missing_ok=True)
                        source.parent.rmdir()
                    except OSError:
                        pass
            if skipped:
                toast(self.window(), f"{len(skipped)} fichier(s) ignoré(s) (format non pris en charge).",
                      "neutral")

        def on_failed(message: str) -> None:
            self.import_progress.setVisible(False)
            self.import_status.setVisible(False)
            self._generated_notice = ""
            toast(self.window(), f"Import impossible : {message}", "danger")

        worker.progress.connect(on_progress)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()

    # -- fichiers -------------------------------------------------------
    def remove_file(self, stored: str) -> None:
        info = next((f for f in self.project.corpus_files if f["stored"] == stored), None)
        if not info:
            return
        ok, _ = ConfirmDialog.ask(
            self.window(), "Retirer ce fichier ?",
            f"« {info['name']} » sera supprimé du corpus du projet.",
            confirm_text="Retirer", danger=True,
            detail="Le fichier d'origine sur votre disque n'est pas touché.",
        )
        if ok:
            self.project.remove_corpus_file(stored)
            self.refresh()
            self.changed.emit()

    def clear_files(self) -> None:
        if not self.project.corpus_files:
            return
        ok, _ = ConfirmDialog.ask(
            self.window(), "Vider le corpus ?",
            "Tous les fichiers du corpus de ce projet seront retirés. "
            "Les versions déjà entraînées ne sont pas supprimées.",
            confirm_text="Vider le corpus",
        )
        if not ok:
            return
        for info in list(self.project.corpus_files):
            self.project.remove_corpus_file(info["stored"])
        self.refresh()
        self.changed.emit()

    def preview_file(self, stored: str) -> None:
        info = next((f for f in self.project.corpus_files if f["stored"] == stored), None)
        if not info:
            return
        text = corpus.preview_corpus([info], self.project.corpus_dir, max_chars=2500)
        dialog = ConfirmDialog(
            self.window(), info["name"],
            text or "(fichier vide ou illisible)",
            confirm_text="Fermer", danger=False, cancel_text="",
        )
        dialog.exec()

    # -- tokenizer ------------------------------------------------------
    def edit_tokenizer(self) -> None:
        dialog = TokenizerDialog(self.window(), self.project)
        if dialog.exec() != TokenizerDialog.Accepted:
            return
        data = dialog.result_data
        kind_changed = data["tokenizer_kind"] != self.project.tokenizer_kind
        if kind_changed and self.project.versions:
            ok, _ = ConfirmDialog.ask(
                self.window(), "Changer de tokenizer ?",
                "Le vocabulaire va changer : les modèles déjà entraînés dans ce projet ne "
                "pourront plus être utilisés pour discuter (leurs embeddings ne correspondent "
                "plus). Ils restent visibles, mais il faudra de nouvelles versions.",
                confirm_text="Changer quand même",
            )
            if not ok:
                return
        self.project.tokenizer_info["kind"] = data["tokenizer_kind"]
        self.project.tokenizer_info["options"] = data["tokenizer_options"]
        self.project.tokenizer_info["trained"] = False
        self.project.invalidate_tokens()
        self.refresh()
        self.changed.emit()
        toast(self.window(), "Tokenizer mis à jour — relancez l'analyse du corpus.", "accent")

    # -- analyse --------------------------------------------------------
    def analyze(self) -> None:
        if not self.project.corpus_files:
            toast(self.window(), "Ajoutez d'abord des fichiers au corpus.", "danger")
            return
        self.analyze_btn.setEnabled(False)
        self.analysis_progress.setVisible(True)
        self.analysis_status.setVisible(True)
        self.analysis_progress.setValue(0)
        self.analysis_status.setText("Préparation…")

        worker = TokenizeWorker(self.project)
        self._worker = worker

        def on_progress(fraction: float, message: str) -> None:
            self.analysis_progress.setValue(int(fraction * 100))
            self.analysis_status.setText(message)

        def on_done(result: dict) -> None:
            self.analysis_progress.setVisible(False)
            self.analysis_status.setVisible(False)
            self.analyze_btn.setEnabled(True)
            self.refresh()
            self.changed.emit()
            toast(
                self.window(),
                f"Corpus analysé : {result['tokens']:,} tokens · vocabulaire "
                f"{result['vocab_size']:,}".replace(",", " "),
                "mint",
            )

        def on_failed(message: str) -> None:
            self.analysis_progress.setVisible(False)
            self.analysis_status.setVisible(False)
            self.analyze_btn.setEnabled(True)
            toast(self.window(), f"Analyse impossible : {message}", "danger")

        worker.progress.connect(on_progress)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()
