"""Onglet « Entraînement » : choix du point de départ, de l'architecture et des
réglages, puis suivi en direct (perte, vitesse, journal)."""

from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import torch

from ...core import gpu, paths, session, sizing
from ...core.model import (
    PRESETS,
    architecture_code,
    architecture_text,
    PRESETS_BY_KEY,
    ModelConfig,
    count_parameters,
    device_label,
    estimate_checkpoint_mb,
    human_params,
    pick_device,
)
from ...core.project import Project, Version, load_settings
from ...core.trainer import TrainConfig
from .. import theme
from ..widgets import (
    Card,
    Chip,
    Collapsible,
    FieldRow,
    IconButton,
    LossChart,
    OptionCard,
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
from ..workers import TrainWorker

ESTIMATE_NOTE = ("Durée indicative : dépend fortement de votre processeur, du gabarit et de la "
                 "longueur de contexte. La vitesse réelle est affichée pendant l'entraînement.")


def estimate_tokens_per_second(params: int, device: torch.device) -> float:
    """Estimation du débit (tokens/s), calibrée sur un CPU récent.

    Volontairement approximative : la vitesse réelle est mesurée pendant
    l'entraînement et remplace cette estimation.
    """
    cpu_tps = min(18000.0, 7.4e9 / max(params, 1))
    if device.type == "cuda":
        return cpu_tps * 15
    if device.type == "mps":
        return cpu_tps * 4
    return cpu_tps


class TrainTab(QWidget):
    finished_run = Signal(object)     # Version
    open_chat = Signal(object)        # Version
    versions_changed = Signal()

    def __init__(self, project: Project, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.worker: TrainWorker | None = None
        self._resume_version: Version | None = None
        self._last_version: Version | None = None
        self._running = False

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(18)

        # ================= colonne gauche : configuration =================
        left_scroll = ScrollArea()
        left_scroll.setFixedWidth(524)
        left_inner = QWidget()
        left = QVBoxLayout(left_inner)
        left.setContentsMargins(0, 4, 8, 24)
        left.setSpacing(14)

        # -- 1. point de départ -----------------------------------------
        start_card = Card(margins=18, spacing=10)
        start_card.body.addWidget(self._step_title("1", "Point de départ"))
        self.base_combo = QComboBox()
        self.base_combo.currentIndexChanged.connect(self._on_base_changed)
        start_card.body.addWidget(self.base_combo)
        self.base_hint = muted("")
        start_card.body.addWidget(self.base_hint)
        self.reset_weights = QCheckBox(
            "Réinitialiser les poids (même architecture, mais on repart de zéro)"
        )
        self.reset_weights.setCursor(Qt.PointingHandCursor)
        self.reset_weights.setVisible(False)
        self.reset_weights.toggled.connect(self._apply_lock_state)
        start_card.body.addWidget(self.reset_weights)
        left.addWidget(start_card)

        # -- 2. architecture --------------------------------------------
        arch_card = Card(margins=18, spacing=10)
        arch_card.body.addWidget(self._step_title("2", "Architecture du modèle"))
        self.arch_hint = muted(
            "Le gabarit fixe la taille du réseau : profondeur (couches), têtes d'attention, "
            "largeur et longueur de contexte. Plus il est gros, plus il apprend de nuances — "
            "mais plus il faut de corpus, de temps et de mémoire."
        )
        arch_card.body.addWidget(self.arch_hint)

        self.preset_cards: dict[str, OptionCard] = {}
        grid = QGridLayout()
        grid.setSpacing(10)
        for i, preset in enumerate(PRESETS):
            card = OptionCard(preset["label"], preset["tagline"])
            card.clicked.connect(lambda k=preset["key"]: self._select_preset(k))
            grid.addWidget(card, i // 2, i % 2)
            self.preset_cards[preset["key"]] = card
        arch_card.body.addLayout(grid)

        self.preset_detail = QFrame()
        self.preset_detail.setObjectName("CardSoft")
        detail_layout = QVBoxLayout(self.preset_detail)
        detail_layout.setContentsMargins(14, 12, 14, 12)
        detail_layout.setSpacing(6)
        self.preset_desc = muted("")
        self.preset_meta = dim("")
        detail_layout.addWidget(self.preset_desc)
        detail_layout.addWidget(self.preset_meta)
        arch_row = QHBoxLayout()
        arch_row.setSpacing(6)
        self.arch_summary = dim("")
        arch_row.addWidget(self.arch_summary, 1)
        arch_btn = QPushButton("Voir l'architecture")
        arch_btn.setObjectName("Ghost")
        arch_btn.setCursor(Qt.PointingHandCursor)
        arch_btn.clicked.connect(self._show_architecture)
        arch_row.addWidget(arch_btn)
        detail_layout.addLayout(arch_row)
        arch_card.body.addWidget(self.preset_detail)

        # réglages personnalisés
        self.custom_box = QFrame()
        self.custom_box.setObjectName("CardSoft")
        custom_layout = QVBoxLayout(self.custom_box)
        custom_layout.setContentsMargins(14, 12, 14, 12)
        custom_layout.setSpacing(12)
        self.layers_spin = self._spin(1, 16, 4)
        self.heads_spin = self._spin(1, 16, 4)
        self.width_spin = self._spin(32, 1024, 128, step=32)
        self.ctx_spin = self._spin(32, 1024, 256, step=32)
        for widget in (self.layers_spin, self.heads_spin, self.width_spin, self.ctx_spin):
            widget.valueChanged.connect(self._update_summary)
        custom_layout.addWidget(FieldRow(
            "Couches (profondeur)", self.layers_spin,
            "Nombre de blocs Transformer empilés. Plus de couches = raisonnement plus profond."))
        custom_layout.addWidget(FieldRow(
            "Têtes d'attention", self.heads_spin,
            "Chaque tête peut se concentrer sur un aspect différent du texte."))
        custom_layout.addWidget(FieldRow(
            "Largeur (d_model)", self.width_spin,
            "Taille des vecteurs internes. Doit être divisible par le nombre de têtes."))
        custom_layout.addWidget(FieldRow(
            "Contexte (tokens)", self.ctx_spin,
            "Nombre maximum de tokens que le modèle « voit » à la fois."))
        self.custom_warning = dim("")
        custom_layout.addWidget(self.custom_warning)
        self.custom_box.setVisible(False)
        arch_card.body.addWidget(self.custom_box)
        left.addWidget(arch_card)

        # -- 3. entraînement --------------------------------------------
        train_card = Card(margins=18, spacing=10)
        train_card.body.addWidget(self._step_title("3", "Réglages d'entraînement"))

        self.epochs_spin = self._spin(1, 200, 1)
        self.batch_spin = self._spin(1, 128, 16)
        self.seq_spin = self._spin(32, 1024, 128, step=32)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0005)
        self.lr_spin.setValue(0.001)
        for widget in (self.epochs_spin, self.batch_spin, self.seq_spin, self.lr_spin):
            widget.valueChanged.connect(self._update_summary)

        train_card.body.addWidget(FieldRow(
            "Époques", self.epochs_spin,
            "Une époque = un passage complet sur le corpus. Pour un petit corpus, plusieurs "
            "époques sont nécessaires ; pour un gros corpus, 1 ou 2 suffisent."))
        train_card.body.addWidget(FieldRow(
            "Taille de lot (batch)", self.batch_spin,
            "Nombre de fenêtres traitées ensemble. Plus grand = plus stable mais plus gourmand "
            "en mémoire."))
        batch_hint_row = QHBoxLayout()
        self.batch_hint = dim("")
        batch_hint_row.addWidget(self.batch_hint, 1)
        self.batch_apply_btn = QPushButton("Appliquer un lot prudent")
        self.batch_apply_btn.setObjectName("Ghost")
        self.batch_apply_btn.setCursor(Qt.PointingHandCursor)
        self.batch_apply_btn.setVisible(False)
        self.batch_apply_btn.clicked.connect(self._apply_calibration_batch)
        batch_hint_row.addWidget(self.batch_apply_btn)
        train_card.body.addLayout(batch_hint_row)
        train_card.body.addWidget(FieldRow(
            "Longueur de contexte", self.seq_spin,
            "Tokens vus d'un coup pendant l'apprentissage (≤ contexte du gabarit). "
            "Plus court = plus rapide."))
        train_card.body.addWidget(FieldRow(
            "Taux d'apprentissage", self.lr_spin,
            "Vitesse d'apprentissage. 1e-3 convient aux petits modèles ; baissez à 3e-4 ou "
            "1e-4 pour les gros gabarits (sinon la perte devient instable)."))

        advanced_body = QWidget()
        adv = QVBoxLayout(advanced_body)
        adv.setContentsMargins(0, 6, 0, 0)
        adv.setSpacing(12)
        self.warmup_spin = self._spin(0, 5000, 40)
        self.accum_spin = self._spin(1, 64, 1)
        self.clip_spin = QDoubleSpinBox()
        self.clip_spin.setRange(0.0, 10.0)
        self.clip_spin.setSingleStep(0.1)
        self.clip_spin.setValue(1.0)
        self.wd_spin = QDoubleSpinBox()
        self.wd_spin.setRange(0.0, 0.5)
        self.wd_spin.setSingleStep(0.005)
        self.wd_spin.setDecimals(3)
        self.wd_spin.setValue(0.01)
        self.eval_spin = self._spin(0, 5000, 50, step=10)
        self.ckpt_spin = self._spin(0, 5000, 100, step=50)
        self.threads_spin = self._spin(0, 128, 0)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["Automatique", "Processeur (CPU)", "GPU (CUDA)"])
        report = gpu.device_report()
        if report["available"]:
            self.device_combo.setItemText(
                0, f"Automatique (GPU : {report['name']})"
            )
            self.device_combo.setItemText(
                2, f"GPU (CUDA) — {report['name']}, {report['vram_gb']:.0f} Go"
            )
        else:
            self.device_combo.setItemText(2, "GPU (CUDA) — indisponible (PyTorch CPU)")
        self.device_combo.currentIndexChanged.connect(self._update_summary)

        adv.addWidget(FieldRow("Échauffement (pas)", self.warmup_spin,
                               "Montée progressive du taux d'apprentissage au démarrage."))
        adv.addWidget(FieldRow("Accumulation de gradient", self.accum_spin,
                               "Simule un plus grand lot sans consommer plus de mémoire."))
        adv.addWidget(FieldRow("Coupure du gradient", self.clip_spin,
                               "Évite les dérapages numériques. 1.0 est un bon défaut."))
        adv.addWidget(FieldRow("Weight decay", self.wd_spin,
                               "Régularisation légère contre le sur-apprentissage."))
        adv.addWidget(FieldRow("Évaluation tous les", self.eval_spin,
                               "Fréquence de mesure de la perte de validation."))
        adv.addWidget(FieldRow("Checkpoint tous les", self.ckpt_spin,
                               "Fréquence de sauvegarde : en cas de coupure, la reprise "
                               "repart du dernier checkpoint."))
        self.amp_check = QCheckBox(
            "Précision mixte (GPU) — plus rapide et moins gourmand en mémoire"
        )
        self.amp_check.setChecked(True)
        self.amp_check.setCursor(Qt.PointingHandCursor)
        self.amp_check.setToolTip(
            "Calcule en bfloat16/float16 quand un GPU CUDA est utilisé. "
            "Sans GPU, cette option est ignorée."
        )
        adv.addWidget(self.amp_check)
        self.compile_check = QCheckBox("Compilation torch.compile (expérimental)")
        self.compile_check.setChecked(False)
        self.compile_check.setCursor(Qt.PointingHandCursor)
        self.compile_check.setToolTip(
            "Peut accélérer les étapes après une première compilation lente. "
            "Recommandé avec CUDA ; si le backend manque ou échoue, Genesis revient "
            "automatiquement au mode standard sans perdre le pas."
        )
        adv.addWidget(self.compile_check)
        adv.addWidget(FieldRow("Fils CPU", self.threads_spin,
                               "0 = laisser PyTorch décider. Réduisez si l'ordinateur rame."))
        adv.addWidget(FieldRow("Appareil", self.device_combo,
                               "Le GPU accélère beaucoup si vous en avez un."))
        train_card.body.addWidget(Collapsible("Paramètres avancés", advanced_body))
        left.addWidget(train_card)

        left.addStretch(1)
        left_scroll.setWidget(left_inner)
        root.addWidget(left_scroll, 0)

        # ================= colonne droite : suivi ========================
        right_scroll = ScrollArea()
        right_inner = QWidget()
        right = QVBoxLayout(right_inner)
        right.setContentsMargins(0, 4, 8, 24)
        right.setSpacing(14)

        summary_card = Card(margins=18, spacing=12)
        head = QHBoxLayout()
        head.addWidget(h3("Résumé avant lancement"))
        head.addStretch(1)
        self.device_chip = Chip(device_label(pick_device("auto")), "neutral")
        head.addWidget(self.device_chip)
        summary_card.body.addLayout(head)

        tiles = QHBoxLayout()
        tiles.setSpacing(10)
        self.tile_train_tokens = StatTile("—", "tokens")
        self.tile_steps_epoch = StatTile("—", "pas/époque")
        self.tile_total_steps = StatTile("—", "pas au total")
        self.tile_duration = StatTile("—", "durée estimée", theme.AMBER)
        self.tile_size = StatTile("—", "poids")
        for tile in (self.tile_train_tokens, self.tile_steps_epoch, self.tile_total_steps,
                     self.tile_duration, self.tile_size):
            tiles.addWidget(tile, 1)
        summary_card.body.addLayout(tiles)

        self.summary_warning = muted("")
        summary_card.body.addWidget(self.summary_warning)

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        self.start_btn = QPushButton("  Démarrer l'entraînement")
        self.start_btn.setObjectName("Primary")
        self.start_btn.setIcon(theme.svg_icon("play", "#FFFFFF", 15))
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.clicked.connect(self.start_training)
        buttons.addWidget(self.start_btn)
        self.stop_btn = QPushButton("  Arrêter et sauvegarder")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.setIcon(theme.svg_icon("stop", "#FF8B94", 15))
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setVisible(False)
        self.stop_btn.clicked.connect(self.stop_training)
        buttons.addWidget(self.stop_btn)
        self.chat_btn = QPushButton("  Discuter avec ce modèle")
        self.chat_btn.setIcon(theme.svg_icon("chat", theme.TEXT, 15))
        self.chat_btn.setCursor(Qt.PointingHandCursor)
        self.chat_btn.setVisible(False)
        self.chat_btn.clicked.connect(self._open_last_chat)
        buttons.addWidget(self.chat_btn)
        buttons.addStretch(1)
        summary_card.body.addLayout(buttons)
        summary_card.body.addWidget(dim(ESTIMATE_NOTE))
        right.addWidget(summary_card)

        # dimensionnement (corpus vs taille du modèle)
        self.sizing_card = Card(margins=18, spacing=8)
        sizing_head = QHBoxLayout()
        sizing_head.addWidget(h3("Volume de corpus / modèle"))
        sizing_head.addStretch(1)
        self.sizing_chip = Chip("—", "neutral")
        sizing_head.addWidget(self.sizing_chip)
        self.sizing_card.body.addLayout(sizing_head)
        self.sizing_headline = QLabel("—")
        self.sizing_headline.setStyleSheet("font-weight: 650; font-size: 13px;")
        self.sizing_card.body.addWidget(self.sizing_headline)
        self.sizing_detail = muted("")
        self.sizing_card.body.addWidget(self.sizing_detail)
        self.sizing_warning = dim("")
        self.sizing_card.body.addWidget(self.sizing_warning)
        guide_row = QHBoxLayout()
        guide_row.addStretch(1)
        guide_btn = QPushButton("Guide des tailles")
        guide_btn.setObjectName("Ghost")
        guide_btn.setIcon(theme.svg_icon("info", theme.TEXT_MUTED, 14))
        guide_btn.setCursor(Qt.PointingHandCursor)
        guide_btn.clicked.connect(self._open_sizing_guide)
        guide_row.addWidget(guide_btn)
        self.sizing_card.body.addLayout(guide_row)
        right.addWidget(self.sizing_card)

        # suivi en direct
        live_card = Card(margins=18, spacing=12)
        live_head = QHBoxLayout()
        live_head.addWidget(h3("Suivi en direct"))
        live_head.addStretch(1)
        self.live_chip = Chip("en attente", "neutral")
        live_head.addWidget(self.live_chip)
        live_card.body.addLayout(live_head)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setFixedHeight(8)
        live_card.body.addWidget(self.progress)

        chips_row = QHBoxLayout()
        chips_row.setSpacing(8)
        self.chip_step = Chip("pas —", "neutral")
        self.chip_loss = Chip("perte —", "accent")
        self.chip_eval = Chip("validation —", "mint")
        self.chip_speed = Chip("— tokens/s", "neutral")
        self.chip_eta = Chip("reste —", "amber")
        for chip in (self.chip_step, self.chip_loss, self.chip_eval, self.chip_speed, self.chip_eta):
            chips_row.addWidget(chip)
        chips_row.addStretch(1)
        live_card.body.addLayout(chips_row)

        self.chart = LossChart()
        self.chart.setMinimumHeight(150)
        live_card.body.addWidget(self.chart, 1)

        log_head = QHBoxLayout()
        log_head.addWidget(dim("Journal"))
        log_head.addStretch(1)
        clear_log = IconButton("trash", "Effacer le journal")
        clear_log.clicked.connect(lambda: self.log_view.clear())
        log_head.addWidget(clear_log)
        live_card.body.addLayout(log_head)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(800)
        self.log_view.setFixedHeight(120)
        self.log_view.setStyleSheet(
            f"background: #0A0D13; border: 1px solid {theme.BORDER_SOFT}; border-radius: 10px;"
            f" color: {theme.TEXT_MUTED}; font-family: 'Cascadia Mono','Consolas','DejaVu Sans Mono',"
            " monospace; font-size: 11px; padding: 8px;"
        )
        live_card.body.addWidget(self.log_view)
        right.addWidget(live_card)

        right_scroll.setWidget(right_inner)
        root.addWidget(right_scroll, 1)

        self._load_defaults()
        self.refresh_calibration()
        self.refresh_versions()
        self._update_summary()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _step_title(self, number: str, text: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(9)
        badge = QLabel(number)
        badge.setFixedSize(20, 20)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"background: {theme.ACCENT_SOFT}; color: #C4BCFF; border-radius: 10px;"
            " font-size: 11px; font-weight: 700;"
        )
        layout.addWidget(badge)
        layout.addWidget(h3(text))
        layout.addStretch(1)
        return container

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        return spin

    def _load_defaults(self) -> None:
        self._select_preset("micro")
        self.seq_spin.setValue(128)

    # ------------------------------------------------------------------
    # versions / point de départ
    # ------------------------------------------------------------------
    def refresh_versions(self) -> None:
        current = self.base_combo.currentData()
        self.base_combo.blockSignals(True)
        self.base_combo.clear()
        self.base_combo.addItem("Modèle neuf — partir de zéro", None)
        for version in reversed(self.project.versions):
            label_text = f"{version.name}"
            if version.is_trained:
                label_text += f"  ·  {version.steps:,} pas".replace(",", " ")
            else:
                label_text += "  ·  sans poids"
            if version.status == "interrompue":
                label_text += "  ·  interrompue"
            self.base_combo.addItem(label_text, version.id)
        if current:
            index = self.base_combo.findData(current)
            if index >= 0:
                self.base_combo.setCurrentIndex(index)
        elif not getattr(self, "_auto_base_done", False):
            trained = [v for v in self.project.versions if v.is_trained]
            if trained:
                latest = trained[-1]
                index = self.base_combo.findData(latest.id)
                if index >= 0:
                    self.base_combo.setCurrentIndex(index)
                self._auto_base_done = True
        self.base_combo.blockSignals(False)
        self._apply_lock_state()

    def _base_version(self) -> Version | None:
        version_id = self.base_combo.currentData()
        return self.project.version(version_id) if version_id else None

    def _on_base_changed(self) -> None:
        self._auto_base_done = True
        self._resume_version = None
        self._apply_lock_state()
        self._update_summary()

    def _apply_lock_state(self) -> None:
        """Verrouille l'architecture quand on repart des poids d'une version."""
        base = self._base_version()
        if base is None:
            self.arch_summary.setVisible(True)
            self.preset_detail.setVisible(True)
            self.base_hint.setText(
                "Un modèle neuf : les poids sont initialisés aléatoirement. "
                "L'entraînement part de zéro, et l'architecture ci-dessous est libre."
            )
            self.reset_weights.setVisible(False)
            self.arch_hint.setText(
                "Le gabarit fixe la taille du réseau : profondeur (couches), têtes d'attention, "
                "largeur et longueur de contexte. Plus il est gros, plus il apprend de nuances — "
                "mais plus il faut de corpus, de temps et de mémoire."
            )
            self.arch_hint.setStyleSheet("")
        else:
            has_weights = base.is_trained or bool(base.init_ref)
            self.base_hint.setText(
                f"Les poids de {base.name} sont repris : le modèle démarre avec ce qu'il a déjà "
                "appris, et l'entraînement ajoute les nouvelles époques. "
                f"L'architecture de {base.name} est conservée pour rester compatible."
            )
            self.reset_weights.setVisible(has_weights)
            if self.reset_weights.isVisible():
                self.reset_weights.setChecked(self.reset_weights.isChecked())
            cfg = ModelConfig.from_dict({**base.model,
                                         "vocab_size": int(self.project.tokenizer_info.get("vocab_size", 2)) or 2})
            self.arch_hint.setText(
                "Architecture héritée et verrouillée : "
                + architecture_text(cfg)
                + "  Décochez « Réinitialiser les poids » ci-dessus pour repartir d'une "
                "architecture libre."
            )
            self.arch_hint.setStyleSheet("color: #B3A9FF;")
        locked = base is not None and not self.reset_weights.isChecked()
        for card in self.preset_cards.values():
            card.set_locked(locked)
        self.custom_box.setVisible(getattr(self, "_preset_key", "micro") == "custom" and not locked)
        self.arch_summary.setVisible(not locked)
        self.preset_detail.setVisible(not locked)
        self._update_summary()

    def prepare_from_version(self, version: Version, resume_exact: bool = False) -> None:
        """Pré-remplit l'onglet depuis la liste des versions."""
        self.refresh_versions()
        index = self.base_combo.findData(version.id)
        if index >= 0 and not resume_exact:
            self.base_combo.setCurrentIndex(index)
        if resume_exact:
            self._resume_version = version
            self.base_combo.setCurrentIndex(0)
            self.base_hint.setText(
                f"Reprise exacte de {version.name} : poids, optimiseur, pas et perte sont "
                "restaurés depuis le dernier checkpoint. Rien n'est perdu."
            )
            self.reset_weights.setVisible(False)
            self._apply_model_cfg_to_ui(version.model)
            self.epochs_spin.setValue(max(1, self.epochs_spin.value()))
        self._update_summary()

    # ------------------------------------------------------------------
    # gabarits
    # ------------------------------------------------------------------
    def _select_preset(self, key: str) -> None:
        if self._base_version() is not None and not self.reset_weights.isChecked():
            toast(self.window(), "Architecture verrouillée : elle est héritée de la version de "
                                 "départ. Décochez « Réinitialiser les poids » pour la changer.",
                  "neutral")
            return
        self._preset_key = key
        for name, card in self.preset_cards.items():
            card.set_selected(name == key)
        preset = PRESETS_BY_KEY[key]
        self.preset_desc.setText(preset["description"])
        self.preset_meta.setText(
            f"Attendu : {preset['expect']}   ·   Vitesse : {preset['speed']}   ·   "
            f"Mémoire : {preset['vram']}"
        )
        self.custom_box.setVisible(key == "custom")
        if key == "custom":
            self.layers_spin.setValue(int(preset["n_layer"]))
            self.heads_spin.setValue(int(preset["n_head"]))
            self.width_spin.setValue(int(preset["n_embd"]))
            self.ctx_spin.setValue(int(preset["block_size"]))
        else:
            self.ctx_spin.setValue(int(preset["block_size"]))
            self.seq_spin.setMaximum(int(preset["block_size"]))
            if self.seq_spin.value() > int(preset["block_size"]):
                self.seq_spin.setValue(int(preset["block_size"]))
        self._update_summary()

    def _open_sizing_guide(self) -> None:
        from ..dialogs import SizingGuideDialog

        vocab = int(self.project.tokenizer_info.get("vocab_size", 0)) or 2
        SizingGuideDialog(
            self.window(), vocab, getattr(self, "_chars_per_token", 4.0),
            getattr(self, "_advice", None),
        ).exec()

    def refresh_calibration(self) -> None:
        """Rappelle la taille de lot mesurée lors de la calibration GPU."""
        data = (load_settings().get("gpu_calibration") or {})
        if not data:
            self.batch_hint.setText("")
            self.batch_apply_btn.setVisible(False)
            return
        device_ok = data.get("device") == "cuda" and gpu.cuda_available()
        hint = (f"Calibration du {data.get('date', '—')} sur {data.get('device_name', '—')} : "
                f"lot jusqu'à {data.get('max_batch', '—')} "
                f"({data.get('tokens_per_s', 0):,.0f} tokens/s mesurés)".replace(",", " "))
        if not device_ok and data.get("device") == "cuda":
            hint += " — GPU indisponible actuellement, ces valeurs ne s'appliquent pas."
        self.batch_hint.setText(hint)
        self.batch_apply_btn.setVisible(device_ok and int(data.get("max_batch", 0)) > 0)

    def _apply_calibration_batch(self) -> None:
        """Transpose la calibration Micro/128 en suggestion prudente pour le modèle courant."""
        data = load_settings().get("gpu_calibration") or {}
        cfg_dict, error = self._model_config()
        if not data or error or not cfg_dict:
            return
        current = ModelConfig.from_dict(cfg_dict)
        current_params = count_parameters(current)
        measured_params = max(1, int(data.get("params", 0)))
        measured_seq = max(1, int(data.get("seq_len", 128)))
        current_seq = max(1, int(self.seq_spin.value()))
        measured_batch = max(1, int(data.get("max_batch", 1)))
        # activation memory roughly scales with parameter count and sequence squared.
        scale = (measured_params / max(1, current_params)) * (measured_seq / current_seq) ** 2
        recommended = max(1, min(128, int(measured_batch * scale * 0.70)))
        self.batch_spin.setValue(recommended)
        toast(self.window(), f"Lot prudent proposé : {recommended}", "success")

    def _show_architecture(self) -> None:
        from ..dialogs import InfoDialog

        cfg, error = self._model_config()
        if error or cfg is None:
            toast(self.window(), error or "Configuration invalide.", "danger")
            return
        model_cfg = ModelConfig.from_dict(cfg)
        body = (
            f"<p>{architecture_text(model_cfg)}</p>"
            "<h3 style='color:#E9EDF6;'>Ce que cela change concrètement</h3>"
            "<p>Chaque réglage modifie réellement le réseau construit au moment du "
            "lancement :</p>"
            "<ul>"
            f"<li><b>Couches ({model_cfg.n_layer})</b> : nombre de blocs empilés.</li>"
            f"<li><b>Têtes ({model_cfg.n_head})</b> : regards parallèles sur le contexte, "
            f"de {model_cfg.n_embd // max(1, model_cfg.n_head)} dimensions chacun.</li>"
            f"<li><b>Largeur ({model_cfg.n_embd})</b> : taille des vecteurs internes.</li>"
            f"<li><b>Contexte ({model_cfg.block_size} tokens)</b> : longueur maximale vue "
            "d'un coup.</li>"
            f"<li><b>Vocabulaire ({model_cfg.vocab_size})</b> : taille de la table "
            "d'embeddings, fixée par le tokenizer du corpus.</li>"
            "</ul>"
            "<h3 style='color:#E9EDF6;'>Code utilisé</h3>"
            f"<pre style='color:#AEB7C9; font-size:11.5px; background:#0C1016; "
            f"padding:12px; border-radius:10px;'>{architecture_code(model_cfg)}</pre>"
        )
        InfoDialog(self.window(), "Architecture du modèle", body, width=680).exec()

    def _apply_model_cfg_to_ui(self, model_cfg: dict) -> None:
        preset = model_cfg.get("preset", "micro")
        self._preset_key = preset if preset in PRESETS_BY_KEY else "custom"
        for name, card in self.preset_cards.items():
            card.set_selected(name == self._preset_key)
        preset_info = PRESETS_BY_KEY[self._preset_key]
        self.preset_desc.setText(preset_info["description"])
        self.preset_meta.setText(
            f"Attendu : {preset_info['expect']}   ·   Vitesse : {preset_info['speed']}"
        )
        self.layers_spin.setValue(int(model_cfg.get("n_layer", 4)))
        self.heads_spin.setValue(int(model_cfg.get("n_head", 4)))
        self.width_spin.setValue(int(model_cfg.get("n_embd", 128)))
        self.ctx_spin.setValue(int(model_cfg.get("block_size", 256)))
        self.seq_spin.setMaximum(int(model_cfg.get("block_size", 256)))
        self.seq_spin.setValue(min(int(model_cfg.get("block_size", 256)), 128))
        self.custom_box.setVisible(self._preset_key == "custom")

    def _model_config(self) -> tuple[dict | None, str | None]:
        """Renvoie (config du modèle, message d'erreur)."""
        vocab = int(self.project.tokenizer_info.get("vocab_size", 0)) or 2
        base = self._base_version()
        if base is not None and not self.reset_weights.isChecked():
            cfg = dict(base.model)
            cfg["vocab_size"] = vocab
            return cfg, None

        key = getattr(self, "_preset_key", "micro")
        preset = PRESETS_BY_KEY[key]
        overrides = {}
        if key == "custom":
            width = int(self.width_spin.value())
            heads = int(self.heads_spin.value())
            if width % heads != 0:
                return None, ("La largeur doit être divisible par le nombre de têtes "
                              f"({width} / {heads}).")
            overrides = {
                "n_layer": int(self.layers_spin.value()),
                "n_head": heads,
                "n_embd": width,
                "block_size": int(self.ctx_spin.value()),
                "dropout": 0.1,
            }
        cfg = ModelConfig(
            vocab_size=vocab,
            block_size=int(overrides.get("block_size", preset["block_size"])),
            n_layer=int(overrides.get("n_layer", preset["n_layer"])),
            n_head=int(overrides.get("n_head", preset["n_head"])),
            n_embd=int(overrides.get("n_embd", preset["n_embd"])),
            dropout=float(overrides.get("dropout", preset["dropout"])),
        )
        data = cfg.to_dict()
        data["preset"] = key
        data["preset_label"] = preset["label"]
        return data, None

    # ------------------------------------------------------------------
    # résumé / estimations
    # ------------------------------------------------------------------
    def _train_config(self) -> TrainConfig:
        device = ["auto", "cpu", "cuda"][self.device_combo.currentIndex()]
        return TrainConfig(
            epochs=int(self.epochs_spin.value()),
            batch_size=int(self.batch_spin.value()),
            seq_len=int(self.seq_spin.value()),
            lr=float(self.lr_spin.value()),
            warmup=int(self.warmup_spin.value()),
            grad_clip=float(self.clip_spin.value()),
            weight_decay=float(self.wd_spin.value()),
            grad_accum=int(self.accum_spin.value()),
            eval_every=int(self.eval_spin.value()),
            ckpt_every=int(self.ckpt_spin.value()),
            threads=int(self.threads_spin.value()),
            device=device,
            amp=bool(self.amp_check.isChecked()),
            compile_model=bool(self.compile_check.isChecked()),
        )

    def _update_summary(self) -> None:
        tokens = int(self.project.tokens_meta.get("tokens", 0))
        cfg, error = self._model_config()
        if error:
            self.custom_warning.setText("⚠ " + error)
            self.start_btn.setEnabled(False)
            return
        self.custom_warning.setText("")

        model_cfg = ModelConfig.from_dict(cfg)
        self.arch_summary.setText(architecture_text(model_cfg))
        params = count_parameters(model_cfg)
        train_cfg = self._train_config()
        n_train = max(1, int(tokens * 0.98))
        steps_per_epoch = max(1, n_train // train_cfg.tokens_per_step)
        total_steps = max(1, steps_per_epoch * train_cfg.epochs)
        device = pick_device(train_cfg.device)
        tps = estimate_tokens_per_second(params, device)
        seconds = total_steps * train_cfg.tokens_per_step / max(1e-6, tps)

        self.tile_train_tokens.set_value(paths.human_count(n_train) if tokens else "—")
        self.tile_steps_epoch.set_value(f"{steps_per_epoch:,}".replace(",", " "))
        self.tile_total_steps.set_value(f"{total_steps:,}".replace(",", " "))
        self.tile_duration.set_value(paths.human_duration(seconds))
        self.tile_size.set_value(f"{estimate_checkpoint_mb(params):.0f} Mo")
        self.device_chip.setText(device_label(device))
        self.device_label = device_label(device)

        # badges des gabarits
        for key, card in self.preset_cards.items():
            preset = PRESETS_BY_KEY[key]
            if key == "custom":
                card.set_badge("vous réglez tout")
                continue
            p = count_parameters(ModelConfig(
                vocab_size=model_cfg.vocab_size, block_size=int(preset["block_size"]),
                n_layer=int(preset["n_layer"]), n_head=int(preset["n_head"]),
                n_embd=int(preset["n_embd"]),
            ))
            card.set_badge(f"{human_params(p)} paramètres")

        # avertissements
        messages = []
        if not tokens:
            messages.append("⚠ Le corpus n'est pas encore analysé : ouvrez l'onglet Corpus.")
            self.start_btn.setEnabled(False)
        else:
            self.start_btn.setEnabled(not self._running)
        if tokens and tokens < 20_000:
            messages.append(
                f"⚠ Corpus très petit ({tokens:,} tokens). Le modèle risque de répéter des "
                "phrases du corpus : ajoutez des textes ou baissez le gabarit.".replace(",", " ")
            )
        elif 0 < tokens < 300_000:
            messages.append(
                "ℹ Corpus de taille modeste : préférez Nano ou Micro, et plusieurs époques."
            )
        if train_cfg.seq_len > model_cfg.block_size:
            messages.append("⚠ Le contexte d'entraînement dépasse le contexte du gabarit.")
            self.start_btn.setEnabled(False)
        if train_cfg.lr > 3e-3 and params > 5e6:
            messages.append("⚠ Taux d'apprentissage élevé pour ce gabarit : risque d'instabilité.")
        self.summary_warning.setText("\n".join(messages))

        # -- dimensionnement corpus / modèle --
        vocab = int(self.project.tokenizer_info.get("vocab_size", 0)) or 2
        chars_per_token = 4.0
        meta = self.project.tokens_meta
        if meta.get("tokens") and meta.get("chars"):
            chars_per_token = max(1.0, float(meta["chars"]) / float(meta["tokens"]))
        advice = sizing.advise(params, n_train, train_cfg.epochs, chars_per_token)
        self._advice = advice
        self._chars_per_token = chars_per_token
        self.sizing_chip.setText(advice.badge)
        self.sizing_chip.set_tone(advice.tone)
        self.sizing_headline.setText(advice.headline)
        self.sizing_detail.setText(advice.detail)
        self.sizing_warning.setText("\n".join(advice.warnings))
        self.sizing_warning.setVisible(bool(advice.warnings))

        if self._resume_version is not None:
            self.start_btn.setText(f"  Reprendre {self._resume_version.name}")
        elif self._base_version() is not None:
            self.start_btn.setText("  Entraîner une nouvelle version")
        else:
            self.start_btn.setText("  Démarrer l'entraînement")

    # ------------------------------------------------------------------
    # entraînement
    # ------------------------------------------------------------------
    def start_training(self) -> None:
        if self._running:
            return
        if not self.project.tokens_ready:
            toast(self.window(), "Analysez d'abord le corpus (onglet Corpus).", "danger")
            return

        train_cfg = self._train_config()
        cfg, error = self._model_config()
        if error or cfg is None:
            toast(self.window(), error or "Configuration invalide.", "danger")
            return
        model_cfg = ModelConfig.from_dict(cfg)
        if int(self.project.tokenizer_info.get("vocab_size", 0)) != model_cfg.vocab_size:
            toast(self.window(),
                  "Le vocabulaire a changé depuis l'analyse : relancez l'analyse du corpus.",
                  "danger")
            return
        if train_cfg.seq_len > model_cfg.block_size:
            toast(self.window(), "La longueur de contexte dépasse celle du gabarit.", "danger")
            return
        tokens_available = int(self.project.tokens_meta.get("tokens", 0))
        if tokens_available < train_cfg.seq_len + 8:
            toast(self.window(),
                  f"Corpus trop petit ({tokens_available} tokens) pour un contexte de "
                  f"{train_cfg.seq_len} : réduisez le contexte ou ajoutez des textes.",
                  "danger")
            return

        base = self._base_version()
        resume_exact = self._resume_version is not None

        try:
            if resume_exact:
                version = self._resume_version
                version.train = train_cfg.to_dict()
                self.project.update_version(version)
                message = f"Reprise de {version.name}."
            else:
                version, message, transferred = session.build_version(
                    self.project,
                    preset_key=getattr(self, "_preset_key", "micro"),
                    model_overrides={k: cfg[k] for k in ("n_layer", "n_head", "n_embd",
                                                         "block_size", "dropout")},
                    train_cfg=train_cfg,
                    base_version_id=base.id if base else None,
                    transfer_weights=not self.reset_weights.isChecked(),
                    device_pref=train_cfg.device,
                )
        except Exception as exc:
            toast(self.window(), f"Impossible de créer la version : {exc}", "danger")
            return

        self._last_version = version
        self.log_view.clear()
        self.chart.clear()
        self._running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setVisible(True)
        self.chat_btn.setVisible(False)
        self.live_chip.setText("en cours")
        self.live_chip.set_tone("accent")
        self.progress.setValue(0)
        self._run_started = time.time()
        self._live_history = None
        self._append_log(message)

        worker = TrainWorker(self.project, version, train_cfg, resume_exact=resume_exact)
        self.worker = worker
        worker.log.connect(self._append_log)
        worker.step.connect(self._on_step)
        worker.finished_run.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.start()
        self.versions_changed.emit()

    def stop_training(self) -> None:
        if self.worker is not None and self._running:
            self.stop_btn.setEnabled(False)
            self.stop_btn.setText("  Arrêt en cours…")
            self.worker.stop()

    def _append_log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.log_view.moveCursor(QTextCursor.End)

    def _on_step(self, payload: dict) -> None:
        total = max(1, int(payload.get("total_steps", 1)))
        step = int(payload.get("step", 0))
        self.progress.setValue(int(1000 * step / total))
        self.chip_step.setText(f"pas {step:,}/{total:,}".replace(",", " "))
        loss = payload.get("loss")
        if loss is not None:
            self.chip_loss.setText(f"perte {loss:.3f}")
        eval_loss = payload.get("eval_loss")
        if eval_loss is not None:
            self.chip_eval.setText(f"validation {eval_loss:.3f}")
        tps = float(payload.get("tok_per_s", 0) or 0)
        self.chip_speed.setText(f"{paths.human_count(tps)} tokens/s")
        eta = float(payload.get("eta", 0) or 0)
        self.chip_eta.setText(f"reste {paths.human_duration(eta)}")
        if tps > 0:
            self.tile_duration.set_value(paths.human_duration(0) if eta <= 0
                                         else paths.human_duration(
                                             time.time() - self._run_started + eta))

        # mise à jour de la courbe en direct
        if self._last_version is not None:
            rows = getattr(self, "_live_history", None)
            if rows is None:
                rows = []
            rows.append([step, loss, eval_loss] if loss is not None else [step, 0, eval_loss])
            self._live_history = rows
            self.chart.set_history(rows)

    def _on_finished(self, summary: dict) -> None:
        self._running = False
        self.stop_btn.setVisible(False)
        self.stop_btn.setEnabled(True)
        self.stop_btn.setText("  Arrêter et sauvegarder")
        self.start_btn.setEnabled(True)
        self._live_history = None
        self.live_chip.setText("interrompu" if summary.get("interrupted") else "terminé")
        self.live_chip.set_tone("amber" if summary.get("interrupted") else "mint")
        self.progress.setValue(1000 if not summary.get("interrupted") else self.progress.value())
        if self._last_version is not None:
            self._last_version = self.project.version(self._last_version.id) or self._last_version
            self.chart.set_history(self._last_version.history)
            self.chat_btn.setVisible(self._last_version.is_trained)
            self._resume_version = None
        self.refresh_versions()
        self._update_summary()
        self.versions_changed.emit()
        self.finished_run.emit(self._last_version)
        token = "Entraînement terminé" if not summary.get("interrupted") else "Entraînement interrompu"
        toast(self.window(),
              f"{token} · {summary.get('steps', 0):,} pas · perte "
              f"{summary.get('loss'):.3f}".replace(",", " ") if summary.get("loss") is not None
              else token, "mint" if not summary.get("interrupted") else "amber")

    def _on_failed(self, message: str) -> None:
        self._running = False
        self.stop_btn.setVisible(False)
        self.start_btn.setEnabled(True)
        self.live_chip.setText("erreur")
        self.live_chip.set_tone("danger")
        self._append_log("ERREUR : " + message)
        if "out of memory" in message.lower() or "memoire" in message.lower():
            old = int(self.batch_spin.value())
            new = max(1, old // 2)
            self.batch_spin.setValue(new)
            self._update_summary()
            self._append_log(
                f"Mémoire GPU insuffisante avec un lot de {old} : la taille de lot a été "
                f"ramenée à {new}. Réduisez aussi la longueur de contexte si le problème "
                "persiste, ou lancez la calibration dans Réglages."
            )
            toast(self.window(), "Mémoire GPU insuffisante — taille de lot réduite. "
                                 "Relancez l'entraînement.", "danger")
            return
        toast(self.window(), "L'entraînement a échoué — voir le journal.", "danger")

    def _open_last_chat(self) -> None:
        if self._last_version is None:
            return
        version = self.project.version(self._last_version.id)
        if version:
            self.open_chat.emit(version)

    # signaux vers la page projet
    def is_running(self) -> bool:
        return self._running
