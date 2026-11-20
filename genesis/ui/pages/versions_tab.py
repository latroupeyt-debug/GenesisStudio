"""Onglet « Versions » : toutes les versions du modèle d'un projet."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ...core import paths
from ...core.model import ModelConfig, count_parameters, human_params
from ...core.project import Project, Version, pretty_date
from .. import theme
from ..dialogs import ConfirmDialog, VersionDetailsDialog
from ..widgets import (
    Card,
    Chip,
    IconButton,
    ScrollArea,
    Sparkline,
    dim,
    h2,
    h3,
    label,
    muted,
    separator,
    toast,
)

STATUS_TONES = {
    "prête": "mint",
    "interrompue": "amber",
    "jamais entraînée": "neutral",
}


class VersionCard(QFrame):
    chat = Signal(object)
    continue_from = Signal(object)
    resume_exact = Signal(object)
    delete = Signal(object)

    def __init__(self, project: Project, version: Version, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.version = version
        self.setObjectName("Card")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(18)

        left = QVBoxLayout()
        left.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        name = QLabel(version.name)
        name.setStyleSheet("font-size: 17px; font-weight: 700;")
        head.addWidget(name)
        head.addWidget(Chip(version.status, STATUS_TONES.get(version.status, "neutral")))
        head.addWidget(Chip(f"{version.model.get('preset_label', '—')} · "
                            f"{human_params(int(version.model.get('params', 0)))} params", "accent"))
        if version.parent:
            head.addWidget(Chip(f"hérite de {version.parent_name or '—'}"))
        head.addStretch(1)
        left.addLayout(head)

        stats = QHBoxLayout()
        stats.setSpacing(18)
        for caption, value in (
            ("pas", f"{version.steps:,}".replace(",", " ")),
            ("tokens vus", paths.human_count(version.tokens_seen)),
            ("perte", "—" if version.last_loss is None else f"{version.last_loss:.3f}"),
            ("validation", "—" if version.best_loss is None else f"{version.best_loss:.3f}"),
            ("durée", paths.human_duration(version.duration)),
        ):
            box = QVBoxLayout()
            box.setSpacing(0)
            value_label = QLabel(value)
            value_label.setStyleSheet("font-weight: 650; font-size: 13px;")
            caption_label = QLabel(caption)
            caption_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10.5px;")
            box.addWidget(value_label)
            box.addWidget(caption_label)
            stats.addLayout(box)
        stats.addStretch(1)
        left.addLayout(stats)

        meta = QLabel(f"Créée le {pretty_date(version.created)}"
                      + (f" · {version.note}" if version.note else ""))
        meta.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        left.addWidget(meta)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        right.setSpacing(8)
        spark = Sparkline(version.history)
        spark.setFixedWidth(170)
        right.addWidget(spark)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        chat_btn = QPushButton("  Discuter")
        chat_btn.setIcon(theme.svg_icon("chat", theme.TEXT, 15))
        chat_btn.setCursor(Qt.PointingHandCursor)
        chat_btn.setEnabled(version.is_trained)
        chat_btn.setToolTip("Ouvrir une conversation avec cette version"
                            if version.is_trained else "Entraînez d'abord cette version")
        chat_btn.clicked.connect(lambda: self.chat.emit(self.version))
        buttons.addWidget(chat_btn)

        if version.status == "interrompue" and version.is_trained:
            resume_btn = QPushButton("  Reprendre")
            resume_btn.setObjectName("Primary")
            resume_btn.setIcon(theme.svg_icon("play", "#FFFFFF", 14))
            resume_btn.setCursor(Qt.PointingHandCursor)
            resume_btn.setToolTip("Reprendre exactement au pas où l'entraînement s'est arrêté")
            resume_btn.clicked.connect(lambda: self.resume_exact.emit(self.version))
            buttons.addWidget(resume_btn)
        else:
            cont_btn = QPushButton("  Continuer")
            cont_btn.setObjectName("Primary")
            cont_btn.setIcon(theme.svg_icon("bolt", "#FFFFFF", 14))
            cont_btn.setCursor(Qt.PointingHandCursor)
            cont_btn.setToolTip("Créer une nouvelle version à partir de celle-ci "
                               "(les poids acquis sont conservés)")
            cont_btn.clicked.connect(lambda: self.continue_from.emit(self.version))
            buttons.addWidget(cont_btn)

        menu_btn = IconButton("dots", "Plus d'options")
        menu_btn.clicked.connect(self._menu)
        buttons.addWidget(menu_btn)
        right.addLayout(buttons)
        layout.addLayout(right)

    def _menu(self) -> None:
        menu = QMenu(self)
        menu.addAction("Voir les détails", self._details)
        menu.addAction("Ouvrir le dossier", lambda: self.project.open_in_explorer())
        menu.addAction("Ouvrir le journal d'entraînement", self._open_log)
        menu.addSeparator()
        menu.addAction("Supprimer cette version…", lambda: self.delete.emit(self.version))
        menu.exec(self.cursor().pos())

    def _details(self) -> None:
        VersionDetailsDialog(self.window(), self.project, self.version).exec()

    def _open_log(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        path = self.project.log_path(self.version.id)
        if path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            toast(self.window(), "Aucun journal pour cette version.", "neutral")


class VersionsTab(QWidget):
    open_chat = Signal(object)
    continue_from = Signal(object)
    resume_exact = Signal(object)

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(14)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(3)
        titles.addWidget(h3("Versions du modèle"))
        titles.addWidget(dim(
            "Chaque entraînement crée une version. Une version peut être continuée : la "
            "nouvelle version part des poids déjà appris, sans tout refaire."
        ))
        header.addLayout(titles, 1)
        self.count_chip = Chip("0", "neutral")
        header.addWidget(self.count_chip, 0, Qt.AlignTop)
        root.addLayout(header)

        self.scroll = ScrollArea()
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 8, 16)
        self.list_layout.setSpacing(12)
        self.list_layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.list_container)
        root.addWidget(self.scroll, 1)

        self.empty_card = Card(margins=26, spacing=10)
        self.empty_card.body.setAlignment(Qt.AlignCenter)
        icon = QLabel()
        icon.setPixmap(theme.svg_icon("layers", theme.TEXT_DIM, 38).pixmap(38, 38))
        icon.setAlignment(Qt.AlignCenter)
        self.empty_card.body.addWidget(icon)
        title = h3("Aucune version pour l'instant")
        title.setAlignment(Qt.AlignCenter)
        self.empty_card.body.addWidget(title)
        hint = muted(
            "Rendez-vous dans l'onglet Entraînement pour créer la première version de "
            "votre modèle. Elle apparaîtra ici avec sa courbe de perte et ses statistiques."
        )
        hint.setAlignment(Qt.AlignCenter)
        self.empty_card.body.addWidget(hint)
        root.addWidget(self.empty_card)

        self.refresh()

    def refresh(self) -> None:
        while self.list_layout.count():
            item = self.list_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)
                widget.deleteLater()

        versions = list(reversed(self.project.versions))  # la plus récente en haut
        self.count_chip.setText(str(len(versions)))
        if not versions:
            self.scroll.hide()
            self.empty_card.show()
            return
        self.empty_card.hide()
        self.scroll.show()

        names = {v.id: v.name for v in versions}
        for version in versions:
            version.parent_name = names.get(version.parent or "", "?") if version.parent else ""
            card = VersionCard(self.project, version)
            card.chat.connect(self.open_chat.emit)
            card.continue_from.connect(self.continue_from.emit)
            card.resume_exact.connect(self.resume_exact.emit)
            card.delete.connect(self.delete_version)
            self.list_layout.addWidget(card)

    def delete_version(self, version: Version) -> None:
        checkpoints = list(self.project.ckpt_dir(version.id).glob("*.pt"))
        size = sum(p.stat().st_size for p in checkpoints) / (1024 ** 2)
        ok, checked = ConfirmDialog.ask(
            self.window(),
            f"Supprimer la version {version.name} ?",
            "Cette version et tous ses checkpoints seront définitivement supprimés. "
            "Les autres versions du projet ne sont pas affectées.",
            confirm_text="Supprimer la version",
            detail=(f"{len(checkpoints)} fichier(s) de poids · {size:.0f} Mo · "
                    f"{version.steps:,} pas d'entraînement".replace(",", " ")),
            checkbox="Je confirme la suppression définitive de cette version",
        )
        if not (ok and checked):
            if ok and not checked:
                toast(self.window(), "Cochez la case de confirmation pour supprimer.", "neutral")
            return
        self.project.delete_version(version.id)
        self.refresh()
        toast(self.window(), f"Version {version.name} supprimée.", "neutral")
