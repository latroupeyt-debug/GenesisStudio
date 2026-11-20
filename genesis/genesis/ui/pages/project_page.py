"""Page d'un projet : en-tête + onglets Corpus / Versions / Entraînement."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ...core import paths
from ...core.project import Project, Version, pretty_date
from .. import theme
from ..dialogs import ConfirmDialog, InfoDialog
from ..widgets import Chip, IconButton, SegmentedControl, dim, muted, toast
from .corpus_tab import CorpusTab
from .train_tab import TrainTab
from .versions_tab import VersionsTab


class ProjectPage(QWidget):
    back = Signal()
    open_chat_signal = Signal(object, object)   # projet, version
    project_changed = Signal(object)

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 18)
        root.setSpacing(16)

        # ---------------- en-tête ----------------
        header = QHBoxLayout()
        header.setSpacing(12)
        back_btn = IconButton("chevron-left", "Retour aux projets")
        back_btn.clicked.connect(self.back.emit)
        header.addWidget(back_btn)

        avatar = QLabel(project.name[:1].upper() or "G")
        avatar.setFixedSize(42, 42)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(
            f"background: {theme.ACCENT_SOFT}; color: #BFB6FF; border-radius: 12px;"
            " font-size: 18px; font-weight: 700;"
        )
        header.addWidget(avatar)

        titles = QVBoxLayout()
        titles.setSpacing(4)
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        self.name_label = QLabel(project.name)
        self.name_label.setStyleSheet("font-size: 21px; font-weight: 700;")
        name_row.addWidget(self.name_label)
        name_row.addStretch(1)
        titles.addLayout(name_row)
        self.chips_row = QHBoxLayout()
        self.chips_row.setSpacing(6)
        titles.addLayout(self.chips_row)
        header.addLayout(titles, 1)

        menu_btn = IconButton("dots", "Actions du projet")
        menu_btn.clicked.connect(self._project_menu)
        header.addWidget(menu_btn)
        help_btn = IconButton("info", "Aide : comment ça marche ?")
        help_btn.clicked.connect(self.show_help)
        header.addWidget(help_btn)
        root.addLayout(header)

        self.segments = SegmentedControl(["Corpus", "Versions", "Entraînement"])
        self.segments.setFixedWidth(400)
        self.segments.changed.connect(self._on_segment)
        root.addWidget(self.segments, 0, Qt.AlignLeft)

        # ---------------- contenu ----------------
        self.stack = QStackedWidget()
        self.corpus_tab = CorpusTab(project)
        self.versions_tab = VersionsTab(project)
        self.train_tab = TrainTab(project)
        self.stack.addWidget(self.corpus_tab)
        self.stack.addWidget(self.versions_tab)
        self.stack.addWidget(self.train_tab)
        root.addWidget(self.stack, 1)

        # ---------------- câblage ----------------
        self.corpus_tab.changed.connect(self._on_corpus_changed)
        self.versions_tab.open_chat.connect(lambda v: self.open_chat_signal.emit(self.project, v))
        self.versions_tab.continue_from.connect(self._continue_from)
        self.versions_tab.resume_exact.connect(lambda v: self._continue_from(v, resume_exact=True))
        self.train_tab.finished_run.connect(self._on_training_finished)
        self.train_tab.versions_changed.connect(self._refresh_versions)
        self.train_tab.open_chat.connect(lambda v: self.open_chat_signal.emit(self.project, v))

        self.refresh_header()

    # ------------------------------------------------------------------
    def refresh_header(self) -> None:
        summary = self.project.summary()
        while self.chips_row.count():
            item = self.chips_row.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        tokenizer_names = {"char": "Caractères", "word": "Mots", "bpe": "BPE"}
        self.chips_row.addWidget(
            Chip(f"Tokenizer · {tokenizer_names.get(self.project.tokenizer_kind, '?')}", "accent")
        )
        self.chips_row.addWidget(Chip(f"{summary['files']} fichier(s)"))
        if summary["tokens_ready"]:
            self.chips_row.addWidget(
                Chip(f"{paths.human_count(summary['tokens'])} tokens · vocab {summary['vocab']}", "mint")
            )
        else:
            self.chips_row.addWidget(Chip("corpus non analysé", "amber"))
        self.chips_row.addWidget(
            Chip(f"{summary['versions']} version" + ("s" if summary["versions"] > 1 else ""))
        )
        self.chips_row.addStretch(1)
        self.name_label.setText(self.project.name)
        self.setWindowTitle(self.project.name)

    def _on_segment(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        if index == 0:
            self.corpus_tab.refresh()
        elif index == 1:
            self.versions_tab.refresh()
        elif index == 2:
            self.train_tab.refresh_versions()
            self.train_tab._update_summary()

    def _on_corpus_changed(self) -> None:
        self.refresh_header()
        self.train_tab.refresh_versions()
        self.train_tab._update_summary()
        self.project_changed.emit(self.project)

    def _refresh_versions(self) -> None:
        self.versions_tab.refresh()
        self.refresh_header()
        self.project_changed.emit(self.project)

    def _on_training_finished(self, version: Version | None) -> None:
        self._refresh_versions()
        self.train_tab.refresh_versions()
        if version is not None and version.is_trained:
            self.segments.set_index(1)

    def _continue_from(self, version: Version, resume_exact: bool = False) -> None:
        self.segments.set_index(2)
        self.train_tab.refresh_versions()
        self.train_tab.prepare_from_version(version, resume_exact=resume_exact)
        if resume_exact:
            toast(self, f"Reprise de {version.name} prête : vérifiez les époques puis lancez.",
                  "accent")
        else:
            toast(self, f"Nouvelle version à partir de {version.name} : les poids sont conservés.",
                  "accent")

    # ------------------------------------------------------------------
    def _project_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Renommer le projet…", self._rename)
        menu.addAction("Ouvrir le dossier du projet", self.project.open_in_explorer)
        menu.addAction("Voir le corpus converti (aperçu)", self._preview_corpus)
        menu.addSeparator()
        menu.addAction("Aide & bonnes pratiques", self.show_help)
        menu.exec(self.mapToGlobal(self.rect().topRight()))

    def _rename(self) -> None:
        from PySide6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(self, "Renommer le projet", "Nouveau nom :",
                                        text=self.project.name)
        if ok and name.strip():
            self.project.rename(name.strip())
            self.refresh_header()
            self.project_changed.emit(self.project)
            toast(self, "Projet renommé.", "mint")

    def _preview_corpus(self) -> None:
        from ...core import corpus

        text = corpus.preview_corpus(self.project.corpus_files, self.project.corpus_dir,
                                     max_chars=4000) or "(corpus vide)"
        dialog = ConfirmDialog(self.window(), "Aperçu du corpus converti", text,
                               confirm_text="Fermer", danger=False, cancel_text="")
        dialog.setMinimumWidth(760)
        dialog.exec()

    def show_help(self) -> None:
        body = """
        <h3 style="color:#E9EDF6;">Le cycle de travail</h3>
        <ol style="margin-left:-20px;">
          <li><b>Corpus</b> — glissez vos fichiers texte. Les formats structurés (CSV, JSON,
              HTML, EPUB…) sont convertis en texte brut. Le corpus est lu par blocs :
              aucun fichier n'est jamais chargé entièrement en mémoire.</li>
          <li><b>Analyse</b> — le tokenizer apprend le vocabulaire de votre corpus, puis tout le
              texte est converti en un fichier de tokens lu en mémoire virtuelle.</li>
          <li><b>Entraînement</b> — choisissez un gabarit (taille du réseau) et lancez.
              Chaque entraînement crée une <b>version</b>.</li>
          <li><b>Discussion</b> — discutez avec chaque version pour juger de sa qualité.</li>
        </ol>

        <h3 style="color:#E9EDF6;">Continuer sans tout refaire</h3>
        <p>Depuis l'onglet <b>Versions</b>, le bouton <b>Continuer</b> crée une nouvelle version
        qui <b>hérite des poids</b> de la précédente : le modèle ne repart pas de zéro, il
        approfondit ce qu'il sait déjà. Le bouton <b>Reprendre</b> (versions interrompues)
        restaure exactement le pas, l'optimiseur et la courbe de perte.</p>

        <h3 style="color:#E9EDF6;">Choisir un gabarit</h3>
        <p>Nano pour un premier essai, Micro pour un modèle utile, Mini/Small si vous avez
        beaucoup de texte et du temps. La fenêtre de contexte indique combien de tokens le
        modèle voit à la fois.</p>

        <h3 style="color:#E9EDF6;">Pourquoi les réponses sont étranges au début ?</h3>
        <p>Un modèle peu entraîné n'a pas encore appris la syntaxe : il produit des suites de
        caractères. C'est normal. Augmentez les époques (ou reprenez l'entraînement), utilisez
        un corpus plus important et abaisez la température dans la page de discussion.</p>
        """
        InfoDialog(self.window(), "Comment fonctionne Genesis Studio", body).exec()
