"""Boîtes de dialogue de l'application (toutes au thème sombre)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import APP_NAME, APP_VERSION
from ..core import paths
from ..core.model import (
    PRESETS,
    architecture_code,
    architecture_text,
    ModelConfig,
    count_parameters,
    estimate_checkpoint_mb,
    human_params,
    preset_model_config,
)
from ..core.project import Project, Version, pretty_date
from ..core.tokenizers import SPECIAL
from . import theme
from .widgets import (
    Card,
    Chip,
    FieldRow,
    OptionCard,
    ScrollArea,
    dim,
    h2,
    h3,
    label,
    muted,
    separator,
    spacer,
)

TOKENIZER_CHOICES = [
    {
        "kind": "char",
        "title": "Caractères",
        "subtitle": "Le plus simple · vocabulaire minuscule",
        "description": (
            "Chaque caractère devient un token (lettres, espaces, ponctuation). "
            "Vocabulaire d'environ 100 entrées : le modèle apprend très vite, même sur un "
            "petit corpus. En contrepartie, un texte long consomme beaucoup de tokens, donc "
            "moins de texte tient dans le contexte."
        ),
    },
    {
        "kind": "word",
        "title": "Mots",
        "subtitle": "Rapide · 1 mot = 1 token",
        "description": (
            "Chaque mot devient un token. Le modèle progresse vite et produit des phrases "
            "nettes, mais tout mot absent du vocabulaire est perdu. Pratique pour des "
            "corpus répétitifs ou des listes de phrases."
        ),
    },
    {
        "kind": "bpe",
        "title": "BPE (sous-mots)",
        "subtitle": "Recommandé · le compromis des grands modèles",
        "description": (
            "L'algorithme BPE (Byte Pair Encoding) apprend les morceaux de mots les plus "
            "fréquents de VOTRE corpus. Aucun caractère inconnu possible (base octet), "
            "bon compromis entre taille du vocabulaire et longueur du contexte. C'est la "
            "méthode utilisée par GPT."
        ),
    },
]

WORD_VOCAB = [2000, 4000, 8000, 16000, 32000]
BPE_VOCAB = [512, 1024, 2048, 4096, 8192]


class BaseDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, title: str = "", width: int = 560):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(width)
        self.setStyleSheet(theme.STYLESHEET)
        self.setPalette(theme.dark_palette())
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(24, 22, 24, 22)
        self.root.setSpacing(16)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        theme.apply_dark_titlebar(self)

    def add_buttons(self, confirm_text: str = "Valider", cancel_text: str = "Annuler",
                    danger: bool = False) -> QPushButton:
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addStretch(1)
        if cancel_text:
            cancel = QPushButton(cancel_text)
            cancel.clicked.connect(self.reject)
            row.addWidget(cancel)
        confirm = QPushButton(confirm_text)
        confirm.setObjectName("Danger" if danger else "Primary")
        confirm.setDefault(True)
        confirm.clicked.connect(self.accept)
        row.addWidget(confirm)
        self.root.addLayout(row)
        self.confirm_button = confirm
        return confirm


# ---------------------------------------------------------------------------
class ConfirmDialog(BaseDialog):
    def __init__(self, parent: QWidget | None, title: str, message: str,
                 confirm_text: str = "Confirmer", danger: bool = True,
                 detail: str = "", checkbox: str = ""):
        super().__init__(parent, title, width=520)
        head = QHBoxLayout()
        head.setSpacing(12)
        icon = QLabel()
        icon.setPixmap(theme.svg_icon("warn" if danger else "info",
                                      theme.DANGER if danger else theme.ACCENT, 28).pixmap(28, 28))
        icon.setFixedWidth(32)
        icon.setAlignment(Qt.AlignTop)
        head.addWidget(icon)
        text_box = QVBoxLayout()
        text_box.setSpacing(8)
        text_box.addWidget(h2(title))
        body = muted(message)
        body.setTextInteractionFlags(Qt.TextSelectableByMouse)
        text_box.addWidget(body)
        if detail:
            text_box.addWidget(dim(detail))
        head.addLayout(text_box, 1)
        self.root.addLayout(head)

        self.checkbox_widget = None
        if checkbox:
            from PySide6.QtWidgets import QCheckBox

            self.checkbox_widget = QCheckBox(checkbox)
            self.root.addWidget(self.checkbox_widget)

        confirm = self.add_buttons(confirm_text, "Annuler", danger=danger)
        if self.checkbox_widget is not None:
            confirm.setEnabled(False)
            self.checkbox_widget.toggled.connect(confirm.setEnabled)

    @property
    def checked(self) -> bool:
        return bool(self.checkbox_widget and self.checkbox_widget.isChecked())

    @staticmethod
    def ask(parent: QWidget | None, title: str, message: str,
            confirm_text: str = "Supprimer", danger: bool = True,
            detail: str = "", checkbox: str = "") -> tuple[bool, bool]:
        dialog = ConfirmDialog(parent, title, message, confirm_text, danger, detail, checkbox)
        result = dialog.exec()
        return result == QDialog.Accepted, dialog.checked


# ---------------------------------------------------------------------------
class NewProjectDialog(BaseDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, "Nouveau projet", width=640)
        self.setWindowTitle("Nouveau projet")
        self.result_data: dict = {}

        self.root.addWidget(h2("Créer un projet"))
        self.root.addWidget(
            muted("Un projet contient un corpus, un tokenizer et toutes les versions de votre "
                  "modèle. Vous pourrez entraîner, comparer et poursuivre chaque version.")
        )

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Ex. : Poésie française, Code Python, Mes notes…")
        self.name_input.textChanged.connect(self._validate)
        self.root.addWidget(label("Nom du projet", "Muted"))
        self.root.addWidget(self.name_input)

        self.desc_input = QPlainTextEdit()
        self.desc_input.setPlaceholderText("Description (facultatif)")
        self.desc_input.setFixedHeight(64)
        self.root.addWidget(self.desc_input)

        self.root.addWidget(separator())
        self.root.addWidget(h3("Tokenizer"))
        self.root.addWidget(
            muted("Le tokenizer découpe le texte en unités que le modèle apprend. "
                  "Ce choix se fait maintenant : il fixe le vocabulaire du corpus.")
        )

        self.cards: dict[str, OptionCard] = {}
        for choice in TOKENIZER_CHOICES:
            card = OptionCard(choice["title"], choice["subtitle"], choice["description"])
            card.clicked.connect(lambda k=choice["kind"]: self.select(k))
            self.root.addWidget(card)
            self.cards[choice["kind"]] = card

        options_row = QHBoxLayout()
        options_row.setSpacing(10)
        options_row.addWidget(label("Taille du vocabulaire", "Muted"))
        self.vocab_combo = QComboBox()
        options_row.addWidget(self.vocab_combo, 1)
        self.root.addLayout(options_row)
        self.vocab_hint = dim("")
        self.root.addWidget(self.vocab_hint)

        hint = dim("Vous ajouterez vos fichiers juste après la création, ou en un clic avec "
                   "le corpus d'exemple fourni.")
        self.root.addWidget(hint)
        self.root.addStretch(1)
        self.confirm = self.add_buttons("Créer le projet", "Annuler")
        self.select("bpe")
        self._validate()

    def select(self, kind: str) -> None:
        for key, card in self.cards.items():
            card.set_selected(key == kind)
            card.set_expanded(key == kind)
        self._kind = kind
        self.vocab_combo.clear()
        if kind == "char":
            self.vocab_combo.addItems(["Automatique (~100 caractères)"])
            self.vocab_combo.setEnabled(False)
            self.vocab_hint.setText(
                "Le vocabulaire sera exactement l'ensemble des caractères présents dans votre corpus."
            )
        elif kind == "word":
            self.vocab_combo.setEnabled(True)
            self.vocab_combo.addItems([f"{v:,} mots".replace(",", " ") for v in WORD_VOCAB])
            self.vocab_combo.setCurrentIndex(2)
            self.vocab_hint.setText(
                "Plus le vocabulaire est grand, plus le modèle est précis, mais plus il est "
                "long à entraîner. Les mots rares sont remplacés par un token « inconnu »."
            )
        else:
            self.vocab_combo.setEnabled(True)
            self.vocab_combo.addItems([f"{v:,} tokens".replace(",", " ") for v in BPE_VOCAB])
            self.vocab_combo.setCurrentIndex(2)
            self.vocab_hint.setText(
                "1 024–2 048 convient à un petit corpus ; 4 096–8 192 pour un corpus riche. "
                "Chaque token supplémentaire agrandit légèrement le modèle."
            )

    def _validate(self) -> None:
        if hasattr(self, "confirm"):
            self.confirm.setEnabled(bool(self.name_input.text().strip()))

    def accept(self) -> None:
        kind = getattr(self, "_kind", "bpe")
        options: dict = {}
        if kind == "word":
            options = {"max_vocab": WORD_VOCAB[max(0, self.vocab_combo.currentIndex())]}
        elif kind == "bpe":
            options = {"vocab_size": BPE_VOCAB[max(0, self.vocab_combo.currentIndex())]}
        self.result_data = {
            "name": self.name_input.text().strip(),
            "description": self.desc_input.toPlainText().strip(),
            "tokenizer_kind": kind,
            "tokenizer_options": options,
        }
        super().accept()


# ---------------------------------------------------------------------------
class TokenizerDialog(BaseDialog):
    """Modification du tokenizer d'un projet existant."""

    def __init__(self, parent: QWidget | None, project: Project):
        super().__init__(parent, "Réglages du tokenizer", width=600)
        self.project = project
        self.result_data: dict = {}
        info = project.tokenizer_info

        self.root.addWidget(h2("Tokenizer du projet"))
        self.root.addWidget(muted(
            "Changer le tokenizer (ou la taille de son vocabulaire) oblige à ré-analyser le "
            "corpus, et rend les modèles déjà entraînés incompatibles : leurs embeddings ne "
            "correspondront plus au vocabulaire."
        ))
        self.root.addWidget(separator())

        self.cards: dict[str, OptionCard] = {}
        for choice in TOKENIZER_CHOICES:
            card = OptionCard(choice["title"], choice["subtitle"], choice["description"])
            card.clicked.connect(lambda k=choice["kind"]: self.select(k))
            self.root.addWidget(card)
            self.cards[choice["kind"]] = card

        row = QHBoxLayout()
        row.addWidget(label("Taille du vocabulaire", "Muted"))
        self.vocab_combo = QComboBox()
        row.addWidget(self.vocab_combo, 1)
        self.root.addLayout(row)
        self.warning = dim("")
        self.root.addWidget(self.warning)
        self.root.addStretch(1)
        self.add_buttons("Appliquer", "Annuler")

        self.select(info.get("kind", "char"))
        current = info.get("options", {})
        if info.get("kind") == "word" and "max_vocab" in current:
            idx = WORD_VOCAB.index(current["max_vocab"]) if current["max_vocab"] in WORD_VOCAB else 2
            self.vocab_combo.setCurrentIndex(idx)
        if info.get("kind") == "bpe" and "vocab_size" in current:
            idx = BPE_VOCAB.index(current["vocab_size"]) if current["vocab_size"] in BPE_VOCAB else 2
            self.vocab_combo.setCurrentIndex(idx)

    def select(self, kind: str) -> None:
        for key, card in self.cards.items():
            card.set_selected(key == kind)
            card.set_expanded(key == kind)
        self._kind = kind
        self.vocab_combo.clear()
        if kind == "char":
            self.vocab_combo.addItems(["Automatique (~100 caractères)"])
            self.vocab_combo.setEnabled(False)
        elif kind == "word":
            self.vocab_combo.setEnabled(True)
            self.vocab_combo.addItems([f"{v:,} mots".replace(",", " ") for v in WORD_VOCAB])
            self.vocab_combo.setCurrentIndex(2)
        else:
            self.vocab_combo.setEnabled(True)
            self.vocab_combo.addItems([f"{v:,} tokens".replace(",", " ") for v in BPE_VOCAB])
            self.vocab_combo.setCurrentIndex(2)
        changed = kind != self.project.tokenizer_info.get("kind")
        self.warning.setText(
            "⚠ Le type de tokenizer change : le corpus sera ré-analysé."
            if changed else "Le corpus sera ré-analysé avec ces réglages."
        )

    def accept(self) -> None:
        kind = self._kind
        options: dict = {}
        if kind == "word":
            options = {"max_vocab": WORD_VOCAB[max(0, self.vocab_combo.currentIndex())]}
        elif kind == "bpe":
            options = {"vocab_size": BPE_VOCAB[max(0, self.vocab_combo.currentIndex())]}
        self.result_data = {"tokenizer_kind": kind, "tokenizer_options": options}
        super().accept()


# ---------------------------------------------------------------------------
class InfoDialog(BaseDialog):
    def __init__(self, parent: QWidget | None, title: str, body_html: str, width: int = 620):
        super().__init__(parent, title, width=width)
        self.root.addWidget(h2(title))
        area = ScrollArea()
        area.setMinimumHeight(280)
        area.setMaximumHeight(520)
        inner = QVBoxLayout(area.container)
        inner.setContentsMargins(0, 0, 8, 0)
        text = QLabel(body_html)
        text.setWordWrap(True)
        text.setTextFormat(Qt.RichText)
        text.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12.5px; line-height: 150%;")
        text.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.LinksAccessibleByMouse)
        text.setOpenExternalLinks(True)
        inner.addWidget(text)
        inner.addStretch(1)
        self.root.addWidget(area)
        self.add_buttons("Fermer", "")


class VersionDetailsDialog(BaseDialog):
    def __init__(self, parent: QWidget | None, project: Project, version: Version):
        super().__init__(parent, f"Détails de {version.name}", width=620)
        self.root.addWidget(h2(f"Version {version.name}"))
        self.root.addWidget(dim(f"Créée le {pretty_date(version.created)}"
                                + (f" · hérite de {version.parent}" if version.parent else "")))

        cfg = ModelConfig.from_dict(version.model)
        params = int(version.model.get("params", 0)) or count_parameters(cfg)
        rows = [
            ("Gabarit", version.model.get("preset_label", "—")),
            ("Paramètres", f"{params:,}".replace(",", " ") + f"  ({human_params(params)})"),
            ("Couches / têtes", f"{cfg.n_layer} / {cfg.n_head}"),
            ("Largeur (d_model)", str(cfg.n_embd)),
            ("Contexte max", f"{cfg.block_size} tokens"),
            ("Vocabulaire", f"{cfg.vocab_size} tokens"),
            ("Pas effectués", f"{version.steps:,}".replace(",", " ")),
            ("Tokens vus (cumulés)", f"{version.tokens_seen:,}".replace(",", " ")),
            ("Perte finale", "—" if version.last_loss is None else f"{version.last_loss:.4f}"),
            ("Meilleure perte (validation)", "—" if version.best_loss is None else f"{version.best_loss:.4f}"),
            ("Temps d'entraînement", f"{version.duration / 60:.1f} min"),
            ("Appareil", version.device),
            ("Statut", version.status),
            ("Dossier", str(project.version_dir(version.id))),
        ]
        card = Card(soft=True, margins=16, spacing=6)
        for key, value in rows:
            line = QHBoxLayout()
            key_label = label(key, "Muted")
            key_label.setFixedWidth(190)
            value_label = QLabel(str(value))
            value_label.setWordWrap(True)
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            line.addWidget(key_label)
            line.addWidget(value_label, 1)
            card.body.addLayout(line)
        self.root.addWidget(card)

        ckpt_dir = project.ckpt_dir(version.id)
        files = sorted(p.name for p in ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
        self.root.addWidget(dim("Checkpoints : " + (", ".join(files) if files else "aucun")))
        self.root.addWidget(dim(architecture_text(cfg)))

        from PySide6.QtWidgets import QPlainTextEdit

        code = QPlainTextEdit(architecture_code(cfg))
        code.setReadOnly(True)
        code.setFixedHeight(168)
        code.setStyleSheet(
            "background:#0C1016; border:1px solid #1B212C; border-radius:10px;"
            "font-family:'Cascadia Mono','Consolas','DejaVu Sans Mono',monospace;"
            "font-size:11px; color:#AEB7C9; padding:8px;"
        )
        self.root.addWidget(code)
        self.add_buttons("Fermer", "")


class AboutDialog(BaseDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent, f"À propos de {APP_NAME}", width=560)
        self.root.addWidget(h2(f"{APP_NAME} {APP_VERSION}"))
        self.root.addWidget(muted(
            "Un atelier pour créer de vrais modèles de langage depuis zéro : votre corpus, "
            "votre tokenizer, votre architecture, vos versions. Tout l'entraînement se passe "
            "sur votre machine, aucune donnée ne quitte votre ordinateur."
        ))
        self.root.addWidget(separator())
        for key, value in (
            ("Données", str(paths.home())),
            ("Espace libre", f"{paths.free_space_gb():.1f} Go"),
            ("Moteur", "PyTorch · GPT écrit à la main"),
        ):
            line = QHBoxLayout()
            k = label(key, "Muted")
            k.setFixedWidth(120)
            line.addWidget(k)
            v = QLabel(value)
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            line.addWidget(v, 1)
            self.root.addLayout(line)
        self.add_buttons("Fermer", "")


# ---------------------------------------------------------------------------
class SettingsDialog(BaseDialog):
    """Réglages, état du matériel (GPU/CUDA) et calibration des performances."""

    def __init__(self, parent: QWidget | None, settings: dict):
        super().__init__(parent, "Réglages & performances", width=660)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(14)
        area = ScrollArea()
        area.setMaximumHeight(620)
        area.setWidget(content)
        self.root.addWidget(area, 1)
        import torch

        from ..core import gpu

        self.settings = settings
        self.worker = None
        report = gpu.device_report()

        content_layout.addWidget(h2("Réglages & performances"))

        # -- matériel ----------------------------------------------------
        card = Card(soft=True, margins=16, spacing=8)
        card.body.addWidget(h3("Appareil de calcul"))
        rows = [
            ("PyTorch", report["torch"]),
            ("Compilé avec CUDA", "oui — CUDA " + str(report["cuda_version"]) if report["cuda_build"] else "non (version CPU)"),
            ("GPU détecté", report["name"] or "aucun"),
            ("Mémoire vidéo", f"{report['vram_gb']:.1f} Go" if report["available"] else "—"),
            ("Capacité de calcul", report["capability"] or "—"),
            ("bfloat16 / TF32", ("oui / oui" if report["bf16"] else "float16 / non") if report["available"] else "—"),
            ("Attention SDPA", "oui — backend PyTorch auto (Flash si compatible)" if report["sdpa"] else "non — attention manuelle"),
            ("torch.compile", "disponible (optionnel)" if report["torch_compile"] else "indisponible"),
        ]
        for key, value in rows:
            line = QHBoxLayout()
            key_label = label(key, "Muted")
            key_label.setFixedWidth(180)
            line.addWidget(key_label)
            value_label = QLabel(str(value))
            value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            line.addWidget(value_label, 1)
            card.body.addLayout(line)

        if not report["available"]:
            warning = QLabel(
                "Aucun GPU CUDA utilisable. Si votre carte est NVIDIA, installez la version "
                "CUDA de PyTorch :\n" + gpu.install_hint()
            )
            warning.setWordWrap(True)
            warning.setTextInteractionFlags(Qt.TextSelectableByMouse)
            warning.setStyleSheet(
                f"color: {theme.AMBER}; background: rgba(255,180,61,0.10);"
                " border: 1px solid #4A3A1A; border-radius: 10px; padding: 10px; font-size: 12px;"
            )
            card.body.addWidget(warning)
            copy_btn = QPushButton("  Copier la commande d'installation")
            copy_btn.setObjectName("Ghost")
            copy_btn.setIcon(theme.svg_icon("copy", theme.TEXT_MUTED, 14))
            copy_btn.clicked.connect(
                lambda: QGuiApplication.clipboard().setText(
                    "pip install torch --index-url https://download.pytorch.org/whl/cu126")
            )
            card.body.addWidget(copy_btn)
        content_layout.addWidget(card)

        # -- mises à jour différentielles --------------------------------
        updates = Card(soft=True, margins=16, spacing=9)
        updates.body.addWidget(h3("Mises à jour"))
        from ..core.updater import UPDATE_REPOSITORY
        self.update_status = dim("")
        self.update_status.setWordWrap(True)
        frozen = bool(getattr(sys, "frozen", False))
        windows = sys.platform == "win32"
        if frozen and windows:
            self.update_status.setText(
                f"Dépôt : github.com/{UPDATE_REPOSITORY}. Les projets et données restent séparés ; "
                "seuls les fichiers modifiés de l'application seront téléchargés."
            )
        else:
            self.update_status.setText(
                "Mise à jour automatique disponible dans l'exécutable Windows. "
                "Depuis les sources, mettez le code à jour avec Git puis relancez."
            )
        updates.body.addWidget(self.update_status)
        self.update_progress = QProgressBar()
        self.update_progress.setRange(0, 100)
        self.update_progress.setValue(0)
        self.update_progress.setFixedHeight(7)
        self.update_progress.setVisible(False)
        updates.body.addWidget(self.update_progress)
        update_buttons = QHBoxLayout()
        update_buttons.addStretch(1)
        self.update_check_btn = QPushButton("  Vérifier les mises à jour")
        self.update_check_btn.setObjectName("Ghost")
        self.update_check_btn.setIcon(theme.svg_icon("refresh", theme.TEXT_MUTED, 14))
        self.update_check_btn.setEnabled(frozen and windows)
        self.update_check_btn.clicked.connect(self._check_for_updates)
        update_buttons.addWidget(self.update_check_btn)
        self.update_install_btn = QPushButton("  Télécharger et installer")
        self.update_install_btn.setObjectName("Primary")
        self.update_install_btn.setIcon(theme.svg_icon("download", "#FFFFFF", 14))
        self.update_install_btn.setVisible(False)
        self.update_install_btn.clicked.connect(self._download_updates)
        update_buttons.addWidget(self.update_install_btn)
        self.update_stop_btn = QPushButton("Annuler")
        self.update_stop_btn.setObjectName("Ghost")
        self.update_stop_btn.setVisible(False)
        self.update_stop_btn.clicked.connect(self._stop_update_download)
        update_buttons.addWidget(self.update_stop_btn)
        updates.body.addLayout(update_buttons)
        content_layout.addWidget(updates)
        self._update_plan = None
        self._update_patches = []
        self.update_worker = None
        self.update_check_worker = None

        # -- calibration -------------------------------------------------
        calib = Card(soft=True, margins=16, spacing=10)
        calib.body.addWidget(h3("Calibration"))
        calib.body.addWidget(muted(
            "Mesure sur votre machine : quelle taille de lot passe au test et à quelle "
            "vitesse le modèle apprend réellement (test plafonné à 128). Le résultat est enregistré et rappelé dans "
            "l'onglet Entraînement."
        ))
        self.calib_label = dim(self._calibration_text())
        calib.body.addWidget(self.calib_label)
        self.calib_bar = QProgressBar()
        self.calib_bar.setRange(0, 100)
        self.calib_bar.setValue(0)
        self.calib_bar.setVisible(False)
        self.calib_bar.setFixedHeight(7)
        calib.body.addWidget(self.calib_bar)
        self.calib_result = muted("")
        self.calib_result.setVisible(False)
        calib.body.addWidget(self.calib_result)
        self.calib_btn = QPushButton("  Calibrer maintenant")
        self.calib_btn.setObjectName("Primary")
        self.calib_btn.setIcon(theme.svg_icon("bolt", "#FFFFFF", 15))
        self.calib_btn.clicked.connect(self.start_calibration)
        calib.body.addWidget(self.calib_btn)
        content_layout.addWidget(calib)

        # -- données -----------------------------------------------------
        data_card = Card(soft=True, margins=16, spacing=8)
        data_card.body.addWidget(h3("Données"))
        path_label = QLabel(str(paths.home()))
        path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        path_label.setWordWrap(True)
        path_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-family: monospace; font-size: 11px;")
        data_card.body.addWidget(path_label)
        data_card.body.addWidget(dim(
            f"Espace libre : {paths.free_space_gb():.1f} Go · sauvegarder = copier ce dossier. "
            "Le mode portable s'active avec un fichier « portable.txt » à côté de l'exécutable."
        ))
        open_btn = QPushButton("  Ouvrir le dossier de données")
        open_btn.setObjectName("Ghost")
        open_btn.setIcon(theme.svg_icon("folder", theme.TEXT_MUTED, 15))
        open_btn.clicked.connect(self._open_data_dir)
        data_card.body.addWidget(open_btn)
        content_layout.addWidget(data_card)

        self.add_buttons("Fermer", "")

    # -- helpers ---------------------------------------------------------
    def _check_for_updates(self) -> None:
        if self.update_check_worker is not None and self.update_check_worker.isRunning():
            return
        from ..core.updater import UPDATE_REPOSITORY
        from ..ui.workers import UpdateCheckWorker

        self.update_check_btn.setEnabled(False)
        self.update_install_btn.setVisible(False)
        self.update_status.setText("Vérification de la dernière release GitHub…")
        worker = UpdateCheckWorker(APP_VERSION, UPDATE_REPOSITORY)
        self.update_check_worker = worker
        worker.done.connect(self._on_update_check_done)
        worker.failed.connect(self._on_update_failed)
        worker.start()

    def _on_update_check_done(self, plan) -> None:
        if self.update_check_worker is not None:
            self.update_check_worker.wait(3000)
        self.update_check_btn.setEnabled(True)
        self._update_plan = plan
        self.update_status.setText(plan.summary)
        try:
            self.update_install_btn.clicked.disconnect()
        except RuntimeError:
            pass
        if plan.available:
            self.update_install_btn.setText("  Télécharger et installer")
            self.update_install_btn.clicked.connect(self._download_updates)
        self.update_install_btn.setVisible(bool(plan.available))
        self.update_install_btn.setEnabled(bool(plan.available))

    def _on_update_failed(self, message: str) -> None:
        if self.update_check_worker is not None:
            self.update_check_worker.wait(3000)
        self.update_check_btn.setEnabled(True)
        self.update_status.setText("Vérification impossible : " + message)

    def _download_updates(self) -> None:
        if not self._update_plan or not self._update_plan.available:
            return
        from ..ui.workers import UpdateDownloadWorker

        self.update_install_btn.setEnabled(False)
        self.update_check_btn.setEnabled(False)
        self.update_progress.setVisible(True)
        self.update_progress.setValue(0)
        self.update_status.setText("Téléchargement puis vérification des patches…")
        worker = UpdateDownloadWorker(self._update_plan, paths.home())
        self.update_worker = worker
        self.update_stop_btn.setVisible(True)
        worker.progress.connect(self._on_update_progress)
        worker.done.connect(self._on_update_downloaded)
        worker.failed.connect(self._on_update_failed)
        worker.start()

    def _on_update_progress(self, fraction: float, message: str) -> None:
        self.update_progress.setValue(max(0, min(100, int(fraction * 100))))
        self.update_status.setText(message)

    def _on_update_downloaded(self, patch_paths, manifests) -> None:
        if self.update_worker is not None:
            self.update_worker.wait(3000)
        self._update_patches = list(patch_paths)
        self.update_progress.setValue(100)
        self.update_progress.setVisible(False)
        self.update_stop_btn.setVisible(False)
        self.update_check_btn.setEnabled(True)
        self.update_install_btn.setVisible(True)
        self.update_install_btn.setEnabled(True)
        self.update_install_btn.setText("  Quitter et installer maintenant")
        self.update_status.setText(
            f"Patches vérifiés (SHA-256) : {len(manifests)} étape(s), "
            f"version finale {manifests[-1]['to_version']}. Cliquez pour fermer Genesis Studio, "
            "appliquer la mise à jour avec rollback de sécurité, puis relancer l'application."
        )
        self.update_install_btn.clicked.disconnect()
        self.update_install_btn.clicked.connect(self._launch_updater)

    def _launch_updater(self) -> None:
        import os
        import subprocess
        import sys

        install_dir = Path(sys.executable).resolve().parent
        helper = install_dir / "GenesisUpdater.exe"
        app_exe = Path(sys.executable).resolve()
        if not helper.is_file():
            self.update_status.setText(
                "Updater compagnon introuvable. Téléchargez le ZIP complet depuis la release GitHub."
            )
            self.update_install_btn.setEnabled(False)
            return
        command = [
            str(helper), "--install-dir", str(install_dir),
            "--data-dir", str(paths.home()), "--wait-pid", str(os.getpid()),
            "--from-version", self._update_plan.current_version,
            "--relaunch", str(app_exe),
        ]
        for patch in self._update_patches:
            command.extend(["--patch", str(patch)])
        flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            subprocess.Popen(command, cwd=str(install_dir), close_fds=True, creationflags=flags)
        except Exception as exc:
            self.update_status.setText(f"Impossible de lancer l'outil de mise à jour : {exc}")
            return
        self.update_status.setText("Genesis Studio va se fermer pour appliquer le patch, puis redémarrer.")
        from PySide6.QtWidgets import QApplication

        self.accept()
        QApplication.quit()

    def _stop_update_download(self) -> None:
        if self.update_worker is not None and self.update_worker.isRunning():
            self.update_worker.stop()
            self.update_stop_btn.setEnabled(False)
            self.update_status.setText("Annulation demandée…")

    def _calibration_text(self) -> str:
        data = self.settings.get("gpu_calibration") or {}
        if not data:
            return "Aucune calibration enregistrée."
        return (f"Dernière calibration ({data.get('date', '—')}) : "
                f"{data.get('device_name', '—')} · lot {data.get('max_batch', '—')} · "
                f"{data.get('tokens_per_s', 0):,.0f} tokens/s · "
                f"{data.get('params', 0):,.0f} paramètres".replace(",", " "))

    def _open_data_dir(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.home())))

    def closeEvent(self, event) -> None:
        # Évite de détruire des QThreads pendant leurs traitements si la fenêtre se ferme.
        for worker in (self.worker, self.update_check_worker, self.update_worker):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "stop"):
                    worker.stop()
                worker.wait(30000)
        super().closeEvent(event)

    def start_calibration(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        from ..core.model import preset_model_config
        from ..ui.workers import BenchmarkWorker

        device = ["auto", "cpu", "cuda"][self.settings.get("device_index", 0)] \
            if isinstance(self.settings.get("device_index"), int) else "auto"
        cfg = preset_model_config("micro", 1024)
        self.calib_btn.setEnabled(False)
        self.calib_bar.setVisible(True)
        self.calib_bar.setValue(5)
        self.calib_result.setVisible(True)
        self.calib_result.setText("Mesure en cours…")

        worker = BenchmarkWorker(cfg, device_pref=device, seq_len=128)
        self.worker = worker

        def on_progress(fraction: float, message: str) -> None:
            self.calib_bar.setValue(int(5 + fraction * 90))
            self.calib_result.setText(message)

        def on_done(result) -> None:
            import time as _time

            self.calib_bar.setValue(100)
            self.calib_bar.setVisible(False)
            self.calib_btn.setEnabled(True)
            self.calib_result.setText(result.summary() +
                                      ("\n" + " · ".join(result.notes) if result.notes else ""))
            self.settings["gpu_calibration"] = {
                "date": _time.strftime("%d/%m/%Y %H:%M"),
                "device": result.device,
                "device_name": result.device_name,
                "params": result.params,
                "seq_len": result.seq_len,
                "max_batch": result.max_batch,
                "tokens_per_s": result.tokens_per_s,
                "amp": result.amp,
            }
            from ..core.project import save_settings

            save_settings(self.settings)
            self.calib_label.setText(self._calibration_text())

        def on_failed(message: str) -> None:
            self.calib_bar.setVisible(False)
            self.calib_btn.setEnabled(True)
            self.calib_result.setText(f"Échec de la calibration : {message}")

        worker.progress.connect(on_progress)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()


# ---------------------------------------------------------------------------
class SizingGuideDialog(BaseDialog):
    """Guide « quelle taille de modèle pour quel volume de texte ? »."""

    def __init__(self, parent: QWidget | None, vocab_size: int, chars_per_token: float,
                 advice=None):
        super().__init__(parent, "Guide des tailles", width=720)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(14)
        area = ScrollArea()
        area.setMaximumHeight(560)
        area.setWidget(content)
        self.root.addWidget(area, 1)

        from ..core import sizing

        content_layout.addWidget(h2("Quelle taille de modèle pour quel corpus ?"))
        content_layout.addWidget(muted(
            "Repères très approximatifs, pas de seuil magique : 20 tokens du corpus par "
            "paramètre est un plancher de sélection, tandis que 100 est un objectif prudent "
            "pour un petit GPT maison. Même un grand ratio ne garantit pas la généralisation : "
            "la diversité, la déduplication et la perte de validation comptent beaucoup."
        ))

        rows = sizing.preset_table(vocab_size, chars_per_token)
        header = QHBoxLayout()
        for text, width in (("Gabarit", 110), ("Paramètres", 90), ("Repère bas (~20 tok/param)", 210),
                            ("Prudent (~100 tok/param)", 210)):
            cell = label(text, "Dim")
            cell.setFixedWidth(width)
            header.addWidget(cell)
        content_layout.addLayout(header)
        content_layout.addWidget(separator())
        for row in rows:
            line = QHBoxLayout()
            name = QLabel(row["label"])
            name.setFixedWidth(110)
            name.setStyleSheet("font-weight: 600;")
            params = QLabel(f"{row['params']:,}".replace(",", " "))
            params.setFixedWidth(90)
            minimum = QLabel(f"{row['tokens_min'] / 1e6:.1f} M tokens · {row['mb_min']:.1f} Mo")
            minimum.setFixedWidth(210)
            comfort = QLabel(f"{row['tokens_comfort'] / 1e6:.0f} M tokens · {row['mb_comfort']:.0f} Mo")
            comfort.setFixedWidth(210)
            for widget in (name, params, minimum, comfort):
                line.addWidget(widget)
            line.addStretch(1)
            content_layout.addLayout(line)
        content_layout.addWidget(dim(
            "Estimation ~4 caractères par token (BPE 1 024–4 096) ; les répétitions et "
            "les documents trop similaires diminuent fortement la valeur réelle du corpus. "
            "1 Mo de texte ≈ 250 000 tokens."
        ))
        content_layout.addWidget(separator())

        content_layout.addWidget(h3("Ce qui fait mémoriser au lieu d'apprendre"))
        content_layout.addWidget(muted(
            "• <b>Trop peu de texte divers</b> pour la taille du modèle : la seule façon de "
            "faire baisser la perte est de recopier.<br>"
            "• <b>Trop d'époques</b> sur un petit corpus : répéter ne crée pas de données, cela "
            "renforce le par-cœur. Le signe : la perte d'entraînement s'effondre alors que la "
            "perte de validation stagne.<br>"
            "• <b>Corpus répétitif</b> (mêmes phrases copiées-collées) : l'application compte "
            "les tokens totaux, pas les passages uniques. Variez les sources et dédupliquez.<br>"
            "Repère pratique : quand la validation cesse de baisser alors que l'entraînement "
            "continue de chuter, arrêtez et ajoutez des données. Le ratio tokens/paramètres "
            "est un filtre d'alerte, pas une prédiction de qualité."
        ))

        if advice is not None:
            content_layout.addWidget(separator())
            content_layout.addWidget(h3("Votre configuration actuelle"))
            content_layout.addWidget(muted(
                f"{advice.detail}<br><br>Époques conseillées : "
                f"<b>{advice.recommended_epochs}</b>."
                + ("<br><br>" + "<br>".join(advice.warnings) if advice.warnings else "")
            ))
        self.add_buttons("Fermer", "")


# ---------------------------------------------------------------------------
class CorpusGeneratorDialog(BaseDialog):
    """Assistant de génération de textes pour enrichir le corpus."""

    def __init__(self, parent: QWidget | None, project: Project):
        super().__init__(parent, "Générer des textes pour le corpus", width=680)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 0)
        content_layout.setSpacing(12)
        area = ScrollArea()
        area.setMaximumHeight(620)
        area.setWidget(content)
        self.root.addWidget(area, 1)
        import sys

        from PySide6.QtWidgets import QCheckBox, QSpinBox
        from ..core.generators import provider_choices
        from ..core.generators import secrets as secret_store
        from ..ui.workers import CorpusGenerationWorker

        self.project = project
        self.worker = None
        self.result = None
        self._providers = {provider.key: provider for provider in provider_choices()}
        self._secret_store = secret_store

        content_layout.addWidget(h2("Générer du texte pour le corpus"))
        intro = muted(
            "Créez des textes synthétiques variés qui seront ajoutés comme un nouveau fichier "
            "au corpus. Le corpus et ses fichiers existants ne sont jamais envoyés au fournisseur."
        )
        intro.setWordWrap(True)
        content_layout.addWidget(intro)

        privacy = QLabel(
            "Confidentialité : un service distant reçoit votre consigne, thème, style et "
            "paramètres, pas vos documents. N'entrez aucune donnée personnelle ou secrète. "
            "Les textes générés peuvent contenir des erreurs : vérifiez-les avant entraînement."
        )
        privacy.setWordWrap(True)
        privacy.setStyleSheet(
            f"color: {theme.AMBER}; background: rgba(255,180,61,0.10);"
            " border: 1px solid #4A3A1A; border-radius: 10px; padding: 10px; font-size: 12px;"
        )
        content_layout.addWidget(privacy)
        self.remote_consent = QCheckBox(
            "J'accepte l'envoi du prompt à l'API distante ; quotas, prix et données dépendent du fournisseur. Aucun document du corpus n'est joint."
        )
        self.remote_consent.setVisible(False)
        content_layout.addWidget(self.remote_consent)

        form = Card(soft=True, margins=16, spacing=10)
        self.provider_combo = QComboBox()
        for provider in provider_choices():
            self.provider_combo.addItem(provider.label, provider.key)
        form.body.addWidget(FieldRow("Fournisseur", self.provider_combo,
                                     "Ollama fonctionne en local. Les API cloud ont des quotas et des conditions propres."))
        self.model_edit = QLineEdit()
        self.model_edit.setPlaceholderText("identifiant du modèle")
        form.body.addWidget(FieldRow("Modèle", self.model_edit,
                                     "OpenRouter : openrouter/free ou un identifiant finissant par :free pour interdire toute facturation."))
        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("clé API — elle n'est jamais écrite dans les fichiers du projet")
        form.body.addWidget(FieldRow("Clé API", self.key_edit, "Nécessaire uniquement pour les fournisseurs cloud."))
        self.remember_key = QCheckBox("Mémoriser cette clé avec le chiffrement Windows DPAPI")
        self.remember_key.setVisible(sys.platform == "win32")
        self.remember_key.setToolTip("La clé est chiffrée pour votre compte Windows et stockée hors des dossiers de projet.")
        form.body.addWidget(self.remember_key)
        self.provider_note = dim("")
        self.provider_note.setWordWrap(True)
        form.body.addWidget(self.provider_note)
        content_layout.addWidget(form)

        instructions = Card(soft=True, margins=16, spacing=10)
        self.category_combo = QComboBox()
        for key, text in (("mélange", "Mélange varié"), ("récits", "Récits"),
                          ("dialogues", "Dialogues"), ("articles", "Articles explicatifs"),
                          ("lettres", "Lettres fictives"), ("descriptions", "Descriptions")):
            self.category_combo.addItem(text, key)
        instructions.body.addWidget(FieldRow("Type de texte", self.category_combo))
        self.language_edit = QLineEdit("français")
        instructions.body.addWidget(FieldRow("Langue", self.language_edit))
        self.topic_edit = QLineEdit()
        self.topic_edit.setPlaceholderText("optionnel — ex. petites villes provençales, astronomie amateur")
        instructions.body.addWidget(FieldRow("Thème", self.topic_edit,
                                             "Laissez vide pour demander des thèmes variés au modèle."))
        self.style_edit = QLineEdit()
        self.style_edit.setPlaceholderText("optionnel — ex. ton accessible, phrases courtes")
        instructions.body.addWidget(FieldRow("Style", self.style_edit))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(1000, 5_000_000)
        self.target_spin.setSingleStep(5000)
        self.target_spin.setValue(10_000)
        self.target_spin.setSuffix(" tok")
        self.requests_spin = QSpinBox()
        self.requests_spin.setRange(1, 1000)
        self.requests_spin.setValue(20)
        self.requests_spin.setSuffix(" req.")
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(128, 4096)
        self.max_tokens_spin.setSingleStep(128)
        self.max_tokens_spin.setValue(1024)
        self.max_tokens_spin.setSuffix(" tok/req.")
        instructions.body.addWidget(FieldRow("Volume visé (approx.)", self.target_spin))
        instructions.body.addWidget(FieldRow("Nombre maximal de requêtes", self.requests_spin))
        instructions.body.addWidget(FieldRow("Longueur maximale par réponse", self.max_tokens_spin))
        content_layout.addWidget(instructions)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        content_layout.addWidget(self.progress)
        self.status = dim("")
        self.status.setWordWrap(True)
        content_layout.addWidget(self.status)
        self.stop_btn = QPushButton("Arrêter après la requête en cours")
        self.stop_btn.setObjectName("Ghost")
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        content_layout.addWidget(self.stop_btn)

        confirm = self.add_buttons("Générer les textes", "Fermer")
        confirm.clicked.disconnect()
        confirm.clicked.connect(self.start_generation)
        self.confirm_button = confirm
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self._provider_changed()

    def _provider_changed(self) -> None:
        key = self.provider_combo.currentData()
        provider = self._providers[key]
        self.model_edit.setText(provider.default_model)
        self.model_edit.setEnabled(True)
        self.key_edit.setVisible(provider.needs_key)
        self.key_edit.setEnabled(provider.needs_key)
        self.remember_key.setVisible(provider.needs_key and __import__("sys").platform == "win32")
        if provider.needs_key:
            self.provider_note.setText(provider.note + " Ne collez pas de données privées ; vérifiez quota, prix et politique de données du fournisseur.")
        else:
            self.provider_note.setText(provider.note)
        self.remote_consent.setVisible(provider.needs_key)
        if not provider.needs_key:
            self.remote_consent.setChecked(False)
        saved = self._secret_store.load_api_key(key) if provider.needs_key else ""
        self.key_edit.setText(saved)
        self.remember_key.setChecked(bool(saved))

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.provider_combo, self.model_edit, self.key_edit, self.category_combo,
                       self.language_edit, self.topic_edit, self.style_edit, self.target_spin,
                       self.requests_spin, self.max_tokens_spin, self.remember_key):
            widget.setEnabled(not busy)
        self.progress.setVisible(busy)
        self.stop_btn.setVisible(busy)
        self.stop_btn.setEnabled(busy)
        self.confirm_button.setEnabled(not busy)

    def start_generation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            return
        from ..core.generators import GenerationConfig
        from ..ui.workers import CorpusGenerationWorker

        provider = self.provider_combo.currentData()
        api_key = self.key_edit.text().strip()
        if self._providers[provider].needs_key and not self.remote_consent.isChecked():
            self.status.setText("Confirmez d'abord l'envoi du prompt au fournisseur distant.")
            return
        if self._providers[provider].needs_key and not api_key:
            self.status.setText("Saisissez une clé API pour ce fournisseur.")
            return
        if provider == "openrouter":
            model = self.model_edit.text().strip()
            if model != "openrouter/free" and not model.endswith(":free"):
                self.status.setText("Sécurité : OpenRouter exige openrouter/free ou un modèle finissant par :free.")
                return
        if provider == "ollama" and not self.model_edit.text().strip():
            self.status.setText("Indiquez le nom d'un modèle installé dans Ollama.")
            return
        if self._providers[provider].needs_key:
            try:
                if self.remember_key.isChecked():
                    self._secret_store.save_api_key(provider, api_key)
                else:
                    self._secret_store.delete_api_key(provider)
            except Exception as exc:
                self.status.setText(f"Impossible de gérer la clé de façon sécurisée : {exc}")
                return
        cfg = GenerationConfig(
            provider=provider,
            model=self.model_edit.text().strip(),
            api_key=api_key,
            category=self.category_combo.currentData(),
            language=self.language_edit.text().strip() or "français",
            topic=self.topic_edit.text(),
            style=self.style_edit.text(),
            target_tokens=self.target_spin.value(),
            max_requests=self.requests_spin.value(),
            max_tokens_per_request=self.max_tokens_spin.value(),
        )
        worker = CorpusGenerationWorker(cfg, self.project.folder / ".generated")
        self.worker = worker
        self._set_busy(True)
        self.status.setText("Connexion au fournisseur…")

        def on_progress(fraction: float, message: str) -> None:
            self.progress.setValue(max(0, min(100, int(fraction * 100))))
            self.status.setText(message)

        def on_done(result) -> None:
            self._set_busy(False)
            self.progress.setValue(100)
            self.result = result
            detail = result.summary()
            if result.error:
                detail += "\n" + result.error
            if result.path:
                detail += "\nFichier temporaire : " + result.path.name
            if result.warnings:
                detail += "\nFiltres : " + "; ".join(result.warnings[:4])
            self.status.setText(detail)
            if result.path:
                # Le signal done est émis juste avant la sortie de run(); attend la fin
                # du QThread pour éviter une destruction pendant son teardown.
                worker.wait(3000)
                self.accept()

        def on_failed(message: str) -> None:
            self._set_busy(False)
            self.status.setText("Échec de génération : " + message)

        worker.progress.connect(on_progress)
        worker.done.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()

    def stop_generation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.stop_btn.setEnabled(False)
            self.status.setText("Arrêt demandé ; le client termine ou annule la requête en cours…")

    def reject(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.stop_generation()
            return
        super().reject()

    def closeEvent(self, event) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.stop_generation()
            event.ignore()
            return
        super().closeEvent(event)
