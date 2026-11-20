"""Page d'accueil : la liste des projets."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...core import paths
from ...core.project import Project, ProjectStore, pretty_date
from .. import theme
from ..dialogs import ConfirmDialog, NewProjectDialog
from ..widgets import (
    Card,
    Chip,
    IconButton,
    ScrollArea,
    dim,
    h1,
    h2,
    h3,
    label,
    muted,
    separator,
)


def tokenizer_label(project: Project) -> str:
    kind = project.tokenizer_kind
    return {"char": "Caractères", "word": "Mots", "bpe": "BPE"}.get(kind, kind)


class ProjectCard(QFrame):
    opened = Signal(object)
    actions = Signal(object, str)

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.setObjectName("Card")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(206)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet(
            f"QFrame#Card {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER};"
            " border-radius: 16px; }"
            f"QFrame#Card:hover {{ border-color: #3A3358; background: {theme.SURFACE_2}; }}"
        )

        summary = project.summary()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        head = QHBoxLayout()
        head.setSpacing(10)
        avatar = QLabel(project.name[:1].upper() or "G")
        avatar.setFixedSize(34, 34)
        avatar.setAlignment(Qt.AlignCenter)
        avatar.setStyleSheet(
            f"background: {theme.ACCENT_SOFT}; color: #BFB6FF; border-radius: 10px;"
            " font-size: 15px; font-weight: 700;"
        )
        head.addWidget(avatar)
        name = QLabel(project.name)
        name.setStyleSheet("font-size: 15px; font-weight: 650;")
        name.setWordWrap(True)
        head.addWidget(name, 1)

        menu_btn = IconButton("dots", "Actions du projet")
        menu_btn.clicked.connect(self._show_menu)
        head.addWidget(menu_btn)
        layout.addLayout(head)

        desc = QLabel(project.description or "Aucune description.")
        desc.setWordWrap(True)
        desc.setFixedHeight(36)
        desc.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 12px;")
        layout.addWidget(desc)

        chips = QHBoxLayout()
        chips.setSpacing(6)
        chips.addWidget(Chip(tokenizer_label(project), "accent"))
        chips.addWidget(Chip(f"{summary['versions']} version" + ("s" if summary["versions"] > 1 else "")))
        if summary["tokens_ready"]:
            chips.addWidget(Chip(f"{paths.human_count(summary['tokens'])} tokens", "mint"))
        chips.addStretch(1)
        layout.addLayout(chips)

        layout.addStretch(1)
        footer = QHBoxLayout()
        info = QLabel(f"{paths.human_size(summary['bytes'])} · {pretty_date(project.updated)}")
        info.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        footer.addWidget(info, 1)
        open_btn = QPushButton("Ouvrir")
        open_btn.setObjectName("Ghost")
        open_btn.setCursor(Qt.PointingHandCursor)
        open_btn.clicked.connect(lambda: self.opened.emit(self.project))
        footer.addWidget(open_btn)
        layout.addLayout(footer)

    def _show_menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Ouvrir", lambda: self.opened.emit(self.project))
        menu.addAction("Renommer…", lambda: self.actions.emit(self.project, "rename"))
        menu.addAction("Dupliquer", lambda: self.actions.emit(self.project, "duplicate"))
        menu.addAction("Ouvrir le dossier", lambda: self.actions.emit(self.project, "folder"))
        menu.addSeparator()
        menu.addAction("Supprimer…", lambda: self.actions.emit(self.project, "delete"))
        menu.exec(self.cursor().pos())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.opened.emit(self.project)


class ProjectsPage(QWidget):
    open_project = Signal(object)
    project_created = Signal(object)

    def __init__(self, store: ProjectStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self.filter_text = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(34, 28, 34, 24)
        root.setSpacing(18)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(4)
        titles.addWidget(h1("Vos projets"))
        titles.addWidget(muted(
            "Chaque projet rassemble un corpus, un tokenizer et toutes les versions "
            "entraînées de votre modèle.", wrap=False
        ))
        header.addLayout(titles, 1)
        new_btn = QPushButton("  Nouveau projet")
        new_btn.setObjectName("Primary")
        new_btn.setIcon(theme.svg_icon("plus", "#FFFFFF", 16))
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.clicked.connect(self.create_project)
        header.addWidget(new_btn, 0, Qt.AlignTop)
        root.addLayout(header)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Rechercher un projet…")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(320)
        self.search.textChanged.connect(self._on_filter)
        root.addWidget(self.search)

        self.scroll = ScrollArea()
        self.grid_container = QWidget()
        self.grid = QGridLayout(self.grid_container)
        self.grid.setContentsMargins(0, 0, 8, 0)
        self.grid.setSpacing(16)
        self.grid.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.grid_container)
        root.addWidget(self.scroll, 1)

        self.empty_state = self._build_empty_state()
        root.addWidget(self.empty_state, 1)

        self.refresh()

    # -- états ----------------------------------------------------------
    def _build_empty_state(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(16)
        layout.setAlignment(Qt.AlignCenter)

        card = Card(margins=34, spacing=14)
        card.setMaximumWidth(660)
        card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        card.body.setAlignment(Qt.AlignCenter)

        icon = QLabel()
        icon.setPixmap(theme.svg_icon("spark", theme.ACCENT, 46).pixmap(46, 46))
        icon.setAlignment(Qt.AlignCenter)
        card.body.addWidget(icon)

        title = h2("Créez votre premier modèle")
        title.setAlignment(Qt.AlignCenter)
        card.body.addWidget(title)
        sub = muted(
            "Aucun projet pour l'instant. Un projet = un corpus de textes + un tokenizer + "
            "autant de versions de modèle que vous voulez. Tout reste sur votre ordinateur."
        )
        sub.setAlignment(Qt.AlignCenter)
        card.body.addWidget(sub)

        steps = QHBoxLayout()
        steps.setSpacing(10)
        for number, title_text, detail in (
            ("1", "Corpus", "Glissez vos fichiers texte (TXT, MD, CSV, JSON, EPUB…)"),
            ("2", "Entraînement", "Choisissez un gabarit, le corpus est lu en flux"),
            ("3", "Discussion", "Discutez avec chaque version, puis continuez-la"),
        ):
            step_card = QFrame()
            step_card.setObjectName("CardSoft")
            step_card.setFixedHeight(116)
            step_layout = QVBoxLayout(step_card)
            step_layout.setContentsMargins(14, 12, 14, 12)
            step_layout.setSpacing(4)
            badge = QLabel(number)
            badge.setFixedSize(22, 22)
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                f"background: {theme.ACCENT_SOFT}; color: #C4BCFF; border-radius: 11px;"
                " font-size: 11px; font-weight: 700;"
            )
            step_layout.addWidget(badge)
            step_layout.addWidget(h3(title_text))
            detail_label = dim(detail)
            step_layout.addWidget(detail_label)
            step_layout.addStretch(1)
            steps.addWidget(step_card, 1)
        card.body.addLayout(steps)

        button = QPushButton("Créer un projet")
        button.setObjectName("Primary")
        button.setCursor(Qt.PointingHandCursor)
        button.clicked.connect(self.create_project)
        card.body.addWidget(button, 0, Qt.AlignCenter)
        layout.addWidget(card)
        return wrapper

    # -- données --------------------------------------------------------
    def _on_filter(self, text: str) -> None:
        self.filter_text = text.strip().lower()
        self.refresh()

    def refresh(self) -> None:
        projects = self.store.list_projects()
        visible = [p for p in projects
                   if not self.filter_text or self.filter_text in p.name.lower()
                   or self.filter_text in p.description.lower()]

        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        if not projects:
            self.scroll.hide()
            self.empty_state.show()
            self.search.setVisible(False)
            return
        self.search.setVisible(True)
        self.empty_state.hide()
        self.scroll.show()

        columns = 3
        for index, project in enumerate(visible):
            card = ProjectCard(project)
            card.opened.connect(self.open_project.emit)
            card.actions.connect(self._on_action)
            self.grid.addWidget(card, index // columns, index % columns)
        for column in range(columns):
            self.grid.setColumnStretch(column, 1)
        if not visible:
            hint = QLabel("Aucun projet ne correspond à cette recherche.")
            hint.setStyleSheet(f"color: {theme.TEXT_DIM};")
            self.grid.addWidget(hint, 0, 0)

    # -- actions --------------------------------------------------------
    def create_project(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() != NewProjectDialog.Accepted:
            return
        data = dialog.result_data
        project = self.store.create(data["name"], data["description"], data["tokenizer_kind"])
        project.tokenizer_info["options"] = data["tokenizer_options"]
        project.save()
        self.refresh()
        self.project_created.emit(project)
        self.open_project.emit(project)

    def _on_action(self, project: Project, action: str) -> None:
        if action == "delete":
            ok, _ = ConfirmDialog.ask(
                self,
                "Supprimer ce projet ?",
                f"« {project.name} » et toutes ses versions seront déplacés dans la corbeille "
                "de l'application. Cette action est définitive pour l'application.",
                confirm_text="Supprimer le projet",
                detail=f"Dossier : {project.folder}",
            )
            if ok:
                self.store.delete(project)
                self.refresh()
        elif action == "duplicate":
            self.store.duplicate(project)
            self.refresh()
        elif action == "folder":
            project.open_in_explorer()
        elif action == "rename":
            from PySide6.QtWidgets import QInputDialog

            name, ok = QInputDialog.getText(self, "Renommer le projet", "Nouveau nom :",
                                            text=project.name)
            if ok and name.strip():
                project.rename(name.strip())
                self.refresh()
