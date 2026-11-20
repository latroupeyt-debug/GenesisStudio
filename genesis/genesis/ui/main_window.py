"""Fenêtre principale : barre latérale de navigation + pile de pages."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, APP_VERSION
from ..core import paths
from ..core.project import Project, ProjectStore, load_settings, save_settings
from ..core.model import device_label, pick_device
from . import theme
from .dialogs import AboutDialog, ConfirmDialog, InfoDialog, NewProjectDialog
from .pages.chat_page import ChatPage
from .pages.project_page import ProjectPage
from .pages.projects_page import ProjectsPage
from .widgets import IconButton, ScrollArea, dim, h3, label, toast


class SidebarButton(QPushButton):
    def __init__(self, project: Project, active: bool, parent: QWidget | None = None):
        super().__init__(parent)
        summary = project.summary()
        self.project = project
        self.setCursor(Qt.PointingHandCursor)
        self.setText("")
        self.setCheckable(True)
        self.setChecked(active)
        self.setFixedHeight(46)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 10, 0)
        layout.setSpacing(9)
        dot = QLabel(project.name[:1].upper() or "G")
        dot.setFixedSize(22, 22)
        dot.setAlignment(Qt.AlignCenter)
        dot.setStyleSheet(
            f"background: {theme.ACCENT_SOFT if active else theme.SURFACE_2};"
            f" color: {'#C4BCFF' if active else theme.TEXT_MUTED}; border-radius: 7px;"
            " font-size: 11px; font-weight: 700;"
        )
        layout.addWidget(dot)
        texts = QVBoxLayout()
        texts.setSpacing(0)
        name = QLabel(project.name)
        name.setStyleSheet(
            f"font-size: 12.5px; font-weight: 600; color: {theme.TEXT if active else '#AEB7C9'};"
        )
        texts.addWidget(name)
        info = QLabel(f"{summary['versions']} version(s) · {paths.human_size(summary['bytes'])}")
        info.setStyleSheet(f"font-size: 10px; color: {theme.TEXT_DIM};")
        texts.addWidget(info)
        layout.addLayout(texts, 1)

        self.setStyleSheet(
            "QPushButton { background: transparent; border: none; border-radius: 10px;"
            " text-align: left; padding: 0px; }"
            f"QPushButton:hover {{ background: {theme.SURFACE_2}; }}"
            f"QPushButton:checked {{ background: {theme.SURFACE_2}; }}"
        )


class Sidebar(QFrame):
    new_project = Signal()
    open_project = Signal(object)
    show_projects = Signal()
    show_settings = Signal()
    show_about = Signal()

    def __init__(self, store: ProjectStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store
        self.setObjectName("Sidebar")
        self.setFixedWidth(252)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 18, 14, 14)
        layout.setSpacing(12)

        # logo
        brand = QHBoxLayout()
        brand.setSpacing(10)
        logo = QLabel()
        logo.setPixmap(theme.app_icon().pixmap(30, 30))
        brand.addWidget(logo)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)
        title = QLabel("Genesis")
        title.setStyleSheet("font-size: 15px; font-weight: 750;")
        brand_text.addWidget(title)
        sub = QLabel("Atelier de modèles IA")
        sub.setStyleSheet(f"font-size: 10px; color: {theme.TEXT_DIM};")
        brand_text.addWidget(sub)
        brand.addLayout(brand_text, 1)
        layout.addLayout(brand)

        layout.addSpacing(6)
        new_btn = QPushButton("  Nouveau projet")
        new_btn.setObjectName("Primary")
        new_btn.setIcon(theme.svg_icon("plus", "#FFFFFF", 15))
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.clicked.connect(self.new_project.emit)
        layout.addWidget(new_btn)

        all_btn = QPushButton("  Tous les projets")
        all_btn.setObjectName("Ghost")
        all_btn.setIcon(theme.svg_icon("layers", theme.TEXT_MUTED, 15))
        all_btn.setCursor(Qt.PointingHandCursor)
        all_btn.setStyleSheet("text-align: left;")
        all_btn.clicked.connect(self.show_projects.emit)
        layout.addWidget(all_btn)

        section = QLabel("PROJETS")
        section.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-size: 10px; font-weight: 700; letter-spacing: 1px;"
            " padding-left: 4px;"
        )
        layout.addWidget(section)

        self.scroll = ScrollArea()
        self.list_host = QWidget()
        self.list_layout = QVBoxLayout(self.list_host)
        self.list_layout.setContentsMargins(0, 0, 4, 0)
        self.list_layout.setSpacing(3)
        self.list_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.list_host)
        layout.addWidget(self.scroll, 1)

        self.empty_hint = dim("Aucun projet.\nCréez-en un pour commencer.")
        layout.addWidget(self.empty_hint)

        layout.addWidget(self._separator())

        info = QLabel(f"Données : {paths.home()}")
        info.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 9.5px;")
        info.setWordWrap(True)
        layout.addWidget(info)
        device = QLabel(device_label(pick_device("auto")))
        device.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 9.5px;")
        layout.addWidget(device)

        bottom = QHBoxLayout()
        bottom.setSpacing(4)
        settings_btn = IconButton("gear", "Réglages")
        settings_btn.clicked.connect(self.show_settings.emit)
        bottom.addWidget(settings_btn)
        about_btn = IconButton("info", f"À propos de {APP_NAME} {APP_VERSION}")
        about_btn.clicked.connect(self.show_about.emit)
        bottom.addWidget(about_btn)
        bottom.addStretch(1)
        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 9.5px;")
        bottom.addWidget(version_label)
        layout.addLayout(bottom)

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background: {theme.BORDER_SOFT};")
        return line

    def refresh(self, active_id: str | None = None) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()
        projects = self.store.list_projects()
        self.empty_hint.setVisible(not projects)
        for project in projects:
            button = SidebarButton(project, active=project.id == active_id)
            button.clicked.connect(lambda _=False, p=project: self.open_project.emit(p))
            self.list_layout.addWidget(button)


class MainWindow(QMainWindow):
    def __init__(self, store: ProjectStore | None = None):
        super().__init__()
        self.store = store or ProjectStore()
        self.settings = load_settings()
        self.project_pages: dict[str, ProjectPage] = {}
        self.current_project_id: str | None = None

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(theme.app_icon())
        self.resize(1440, 940)
        self.setMinimumSize(1120, 720)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.sidebar = Sidebar(self.store)
        self.sidebar.new_project.connect(self.new_project)
        self.sidebar.open_project.connect(self.open_project)
        self.sidebar.show_projects.connect(self.show_projects)
        self.sidebar.show_settings.connect(self.show_settings)
        self.sidebar.show_about.connect(lambda: AboutDialog(self).exec())
        layout.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(central)

        self.projects_page = ProjectsPage(self.store)
        self.projects_page.open_project.connect(self.open_project)
        self.projects_page.project_created.connect(lambda p: self.sidebar.refresh(p.id))
        self.stack.addWidget(self.projects_page)

        self.sidebar.refresh()
        self.show_projects()

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        theme.apply_dark_titlebar(self)

    def show_projects(self) -> None:
        self.stack.setCurrentWidget(self.projects_page)
        self.projects_page.refresh()
        self.sidebar.refresh()
        self.current_project_id = None
        self.setWindowTitle(APP_NAME)

    def open_project(self, project: Project) -> None:
        page = self.project_pages.get(project.id)
        if page is None:
            page = ProjectPage(project)
            page.back.connect(self.show_projects)
            page.open_chat_signal.connect(self.open_chat)
            page.project_changed.connect(lambda p: self.sidebar.refresh(p.id))
            self.project_pages[project.id] = page
            self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)
        self.current_project_id = project.id
        self.sidebar.refresh(project.id)
        self.setWindowTitle(f"{project.name} — {APP_NAME}")

    def open_chat(self, project: Project, version) -> None:
        page = ChatPage(project, version)
        page.back.connect(lambda: self._close_chat(project))
        self.stack.addWidget(page)
        self.stack.setCurrentWidget(page)
        self.setWindowTitle(f"Discussion {version.name} — {APP_NAME}")

    def _close_chat(self, project: Project) -> None:
        current = self.stack.currentWidget()
        if isinstance(current, ChatPage):
            self.stack.removeWidget(current)
            current.deleteLater()
        self.open_project(project)

    # ------------------------------------------------------------------
    def new_project(self) -> None:
        dialog = NewProjectDialog(self)
        if dialog.exec() != NewProjectDialog.Accepted:
            return
        data = dialog.result_data
        project = self.store.create(data["name"], data["description"], data["tokenizer_kind"])
        project.tokenizer_info["options"] = data["tokenizer_options"]
        project.save()
        self.open_project(project)
        self.projects_page.refresh()
        toast(self, "Projet créé — ajoutez des fichiers dans l'onglet Corpus.", "mint")

    def show_settings(self) -> None:
        from .dialogs import SettingsDialog

        dialog = SettingsDialog(self, self.settings)
        dialog.exec()
        # la calibration peut avoir changé : on rafraîchit l'onglet d'entraînement
        for page in self.project_pages.values():
            if getattr(page, "train_tab", None) is not None:
                page.train_tab.refresh_calibration()

    def closeEvent(self, event) -> None:  # noqa: N802
        running = [p for p in self.project_pages.values()
                   if getattr(p, "train_tab", None) is not None and p.train_tab.is_running()]
        if running:
            ok, _ = ConfirmDialog.ask(
                self, "Un entraînement est en cours",
                "Un entraînement tourne encore. Fermer maintenant l'interrompt "
                "(le dernier checkpoint est conservé, vous pourrez reprendre plus tard).",
                confirm_text="Fermer quand même",
            )
            if not ok:
                event.ignore()
                return
            for page in running:
                page.train_tab.stop_training()
                if page.train_tab.worker is not None:
                    page.train_tab.worker.wait(4000)
        save_settings(self.settings)
        event.accept()
