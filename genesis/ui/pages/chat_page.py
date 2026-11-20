"""Page de discussion avec une version entraînée du modèle."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QKeyEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...core import chat, paths
from ...core.model import human_params
from ...core.project import Project, Version, pretty_date
from .. import theme
from ..widgets import (
    Card,
    Chip,
    IconButton,
    ScrollArea,
    dim,
    h2,
    h3,
    label,
    muted,
    separator,
    toast,
)
from ..workers import GenerateWorker, ModelLoadWorker


class ChatInput(QPlainTextEdit):
    submitted = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and not (
            event.modifiers() & Qt.ShiftModifier
        ):
            self.submitted.emit()
            return
        super().keyPressEvent(event)


class Bubble(QWidget):
    """Message de la conversation."""

    def __init__(self, text: str, role: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.role = role
        self.raw_text = text
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(0)

        if role == "note":
            # note d'information : pleine largeur, sans bulle
            layout.setContentsMargins(2, 6, 2, 6)
            self.text_label = QLabel(text)
            self.text_label.setWordWrap(True)
            self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.text_label.setStyleSheet(
                f"color: {theme.TEXT_DIM}; font-size: 12.5px; font-style: italic;"
            )
            layout.addWidget(self.text_label, 1)
            self.header = QLabel("")
            self.footer = QLabel("")
            return

        frame = QFrame()
        frame.setMinimumWidth(240)
        frame.setMaximumWidth(780)
        body = QVBoxLayout(frame)
        body.setContentsMargins(15, 11, 15, 11)
        body.setSpacing(5)

        if role == "user":
            frame.setObjectName("BubbleUser")
            frame.setStyleSheet(
                f"QFrame#BubbleUser {{ background: {theme.ACCENT_SOFT};"
                " border: 1px solid #3D356F; border-radius: 14px; }"
            )
            layout.addStretch(1)
            layout.addWidget(frame, 0, Qt.AlignRight)
        else:
            frame.setObjectName("BubbleModel")
            frame.setStyleSheet(
                f"QFrame#BubbleModel {{ background: {theme.SURFACE_2};"
                f" border: 1px solid {theme.BORDER_SOFT}; border-radius: 14px; }}"
            )
            layout.addWidget(frame, 0, Qt.AlignLeft)
            layout.addStretch(1)

        self.header = QLabel({"user": "Vous", "assistant": "Modèle"}.get(role, role))
        self.header.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-size: 10.5px; font-weight: 700;"
            " letter-spacing: 0.4px;"
        )
        body.addWidget(self.header)

        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.text_label.setStyleSheet(f"color: {theme.TEXT}; font-size: 13.5px;")
        body.addWidget(self.text_label)

        self.footer = QLabel("")
        self.footer.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10.5px;")
        self.footer.setVisible(False)
        body.addWidget(self.footer)

    def set_text(self, text: str) -> None:
        self.raw_text = text
        self.text_label.setText(text or "…")

    def append_text(self, piece: str) -> None:
        self.raw_text += piece
        self.text_label.setText(self.raw_text)

    def set_footer(self, text: str) -> None:
        self.footer.setText(text)
        self.footer.setVisible(bool(text))


class ChatPage(QWidget):
    back = Signal()

    def __init__(self, project: Project, version: Version, parent: QWidget | None = None):
        super().__init__(parent)
        self.project = project
        self.version = version
        self.model = None
        self.tokenizer = None
        self.device = None
        self.worker: GenerateWorker | None = None
        self.loader: ModelLoadWorker | None = None
        self.selected = False
        self.current_bubble: Bubble | None = None
        self.turns: list[tuple[str, str]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 20)
        root.setSpacing(14)

        # ---------------- en-tête ----------------
        header = QHBoxLayout()
        header.setSpacing(12)
        back_btn = IconButton("chevron-left", "Retour au projet")
        back_btn.clicked.connect(self._go_back)
        header.addWidget(back_btn)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        title = QLabel(f"Discussion avec {version.name}")
        title.setStyleSheet("font-size: 19px; font-weight: 700;")
        name_row.addWidget(title)
        name_row.addWidget(Chip(version.status, "mint" if version.status == "prête" else "amber"))
        params = int(version.model.get("params", 0))
        if params:
            name_row.addWidget(Chip(human_params(params) + " paramètres", "accent"))
        name_row.addStretch(1)
        titles.addLayout(name_row)
        subtitle = dim(
            f"{project.name} · {version.steps:,} pas d'entraînement · "
            f"{paths.human_count(version.tokens_seen)} tokens vus · "
            f"perte {version.last_loss:.3f}" if version.last_loss is not None
            else f"{project.name} · cette version n'a pas encore été entraînée"
        )
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)

        self.new_btn = QPushButton("  Nouvelle conversation")
        self.new_btn.setObjectName("Ghost")
        self.new_btn.setIcon(theme.svg_icon("refresh", theme.TEXT_MUTED, 15))
        self.new_btn.clicked.connect(self.new_conversation)
        header.addWidget(self.new_btn)
        root.addLayout(header)

        # ---------------- corps ----------------
        body = QHBoxLayout()
        body.setSpacing(16)

        chat_col = QVBoxLayout()
        chat_col.setSpacing(10)
        self.scroll = ScrollArea()
        self.messages_host = QWidget()
        self.messages = QVBoxLayout(self.messages_host)
        self.messages.setContentsMargins(2, 2, 10, 2)
        self.messages.setSpacing(10)
        self.messages.addStretch(1)
        self.scroll.setWidget(self.messages_host)
        chat_col.addWidget(self.scroll, 1)

        self.status_label = dim("Chargement du modèle…")
        chat_col.addWidget(self.status_label)

        input_card = Card(margins=12, spacing=8)
        self.input = ChatInput()
        self.input.setPlaceholderText(
            "Écrivez le début d'un texte, une question, une instruction… "
            "(Entrée pour envoyer, Maj+Entrée pour aller à la ligne)"
        )
        self.input.setFixedHeight(84)
        self.input.submitted.connect(self.send)
        input_card.body.addWidget(self.input)

        input_row = QHBoxLayout()
        input_row.setSpacing(8)
        self.history_toggle = QPushButton("Contexte : question seule")
        self.history_toggle.setObjectName("Chip")
        self.history_toggle.setCheckable(True)
        self.history_toggle.setCursor(Qt.PointingHandCursor)
        self.history_toggle.setToolTip(
            "Désactivé : le modèle ne reçoit que votre dernier message.\n"
            "Activé : les échanges précédents sont ajoutés au contexte "
            "(utile en mode dialogue)."
        )
        self.history_toggle.toggled.connect(
            lambda checked: self.history_toggle.setText(
                "Contexte : conversation" if checked else "Contexte : question seule"
            )
        )
        input_row.addWidget(self.history_toggle)
        copy_btn = QPushButton("  Copier la réponse")
        copy_btn.setObjectName("Chip")
        copy_btn.setIcon(theme.svg_icon("copy", theme.TEXT_MUTED, 13))
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.setToolTip("Copier la dernière réponse du modèle dans le presse-papiers")
        copy_btn.clicked.connect(self.copy_last)
        input_row.addWidget(copy_btn)
        input_row.addStretch(1)
        self.stop_btn = QPushButton("  Arrêter")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.setVisible(False)
        self.stop_btn.setIcon(theme.svg_icon("stop", "#FF8B94", 14))
        self.stop_btn.clicked.connect(self.stop_generation)
        input_row.addWidget(self.stop_btn)
        self.send_btn = QPushButton("  Envoyer")
        self.send_btn.setObjectName("Primary")
        self.send_btn.setIcon(theme.svg_icon("send", "#FFFFFF", 15))
        self.send_btn.setCursor(Qt.PointingHandCursor)
        self.send_btn.clicked.connect(self.send)
        self.send_btn.setEnabled(False)
        input_row.addWidget(self.send_btn)
        input_card.body.addLayout(input_row)
        chat_col.addWidget(input_card)
        body.addLayout(chat_col, 1)

        # ---------------- panneau réglages ----------------
        side_scroll = ScrollArea()
        side_scroll.setFixedWidth(304)
        side_inner = QWidget()
        side = QVBoxLayout(side_inner)
        side.setContentsMargins(0, 0, 8, 12)
        side.setSpacing(14)

        settings_card = Card(margins=16, spacing=12)
        settings_card.body.addWidget(h3("Réglages de génération"))

        settings_card.body.addWidget(label("Gabarit de prompt", "Muted"))
        self.template_combo = QComboBox()
        for key, spec in chat.PROMPT_TEMPLATES.items():
            self.template_combo.addItem(spec["label"], key)
        self.template_combo.currentIndexChanged.connect(self._on_template_changed)
        settings_card.body.addWidget(self.template_combo)
        self.template_hint = dim("")
        settings_card.body.addWidget(self.template_hint)

        settings_card.body.addWidget(separator())

        self.temp_slider, self.temp_value = self._slider_row(
            settings_card, "Température", 0, 150, 80,
            "0 = très sage et répétitif · 0.8 = équilibré · 1.2 = créatif et imprévisible."
        )
        self.topk_spin = QSpinBox()
        self.topk_spin.setRange(0, 200)
        self.topk_spin.setValue(40)
        settings_card.body.addWidget(self._field("top-k", self.topk_spin,
                                                 "Ne tire que parmi les k tokens les plus probables."))
        self.topp_slider, self.topp_value = self._slider_row(
            settings_card, "top-p", 0, 100, 95,
            "Coupe la queue de probabilité : 0.95 garde les tokens les plus vraisemblables."
        )
        self.maxtok_spin = QSpinBox()
        self.maxtok_spin.setRange(8, 2000)
        self.maxtok_spin.setValue(160)
        settings_card.body.addWidget(self._field("Tokens max", self.maxtok_spin,
                                                 "Longueur maximale de la réponse."))
        self.rep_slider, self.rep_value = self._slider_row(
            settings_card, "Anti-répétition", 100, 160, 110,
            "Pénalise les tokens déjà utilisés pour éviter les boucles."
        )
        side.addWidget(settings_card)

        info_card = Card(margins=16, spacing=8, soft=True)
        info_card.body.addWidget(h3("À propos de ce modèle"))
        self.info_label = muted("")
        info_card.body.addWidget(self.info_label)
        side.addWidget(info_card)

        side.addStretch(1)
        side_scroll.setWidget(side_inner)
        body.addWidget(side_scroll, 0)
        root.addLayout(body, 1)

        self._on_template_changed()
        self._empty_state()
        self.load_model()

    # ------------------------------------------------------------------
    def _field(self, title: str, widget: QWidget, hint: str = "") -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        row = QHBoxLayout()
        row.addWidget(label(title, "Muted"))
        row.addStretch(1)
        widget.setFixedWidth(72)
        row.addWidget(widget)
        layout.addLayout(row)
        if hint:
            layout.addWidget(dim(hint))
        return box

    def _slider_row(self, card: Card, title: str, minimum: int, maximum: int,
                    value: int, hint: str) -> tuple[QSlider, QLabel]:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        head = QHBoxLayout()
        head.addWidget(label(title, "Muted"))
        head.addStretch(1)
        value_label = QLabel(f"{value / 100:.2f}")
        value_label.setStyleSheet(f"color: {theme.TEXT}; font-weight: 600; font-size: 12px;")
        head.addWidget(value_label)
        layout.addLayout(head)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)
        slider.valueChanged.connect(lambda v: value_label.setText(f"{v / 100:.2f}"))
        layout.addWidget(slider)
        layout.addWidget(dim(hint))
        card.body.addWidget(box)
        return slider, value_label

    # ------------------------------------------------------------------
    def _go_back(self) -> None:
        self.stop_generation()
        if self.loader is not None and self.loader.isRunning():
            self.loader.wait(2000)
        self.back.emit()

    def _empty_state(self) -> None:
        note = Bubble(
            "Ce modèle est petit et entraîné sur votre corpus : il prolonge le texte plutôt "
            "qu'il ne dialogue, surtout s'il n'a vu que peu d'époques. Commencez par une phrase "
            "dans le style de votre corpus.", "note",
        )
        self.messages.insertWidget(self.messages.count() - 1, note)

    def _on_template_changed(self) -> None:
        key = self.template_combo.currentData()
        spec = chat.PROMPT_TEMPLATES.get(key, {})
        self.template_hint.setText(spec.get("description", ""))
        if key in ("instruction", "dialogue"):
            self.history_toggle.setChecked(True)
        else:
            self.history_toggle.setChecked(False)
        self.history_toggle.setText(
            "Contexte : conversation" if self.history_toggle.isChecked()
            else "Contexte : question seule"
        )

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        self.status_label.setText("Chargement du modèle…")
        self.send_btn.setEnabled(False)
        loader = ModelLoadWorker(self.project, self.version)
        self.loader = loader

        def on_loaded(model, tokenizer, device, meta) -> None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.status_label.setText(
                f"Prêt · checkpoint « {meta.get('checkpoint', '—')} » · "
                f"contexte {model.cfg.block_size} tokens · vocabulaire {model.cfg.vocab_size}"
            )
            self.send_btn.setEnabled(True)
            self.input.setFocus()
            self.info_label.setText(
                f"Entraîné sur « {self.project.name} » avec un tokenizer "
                f"{self._tokenizer_name()}.\n\n"
                f"{self.version.steps:,} pas · {paths.human_count(self.version.tokens_seen)} tokens vus · "
                f"créée le {pretty_date(self.version.created)}.\n\n"
                "Plus le nombre de pas est élevé, plus le texte est fluide. "
                "Température basse = plus fidèle au corpus.".replace(",", " ")
            )

        def on_failed(message: str) -> None:
            self.status_label.setText("Modèle indisponible")
            note = Bubble(f"Impossible de charger ce modèle : {message}", "note")
            self.messages.insertWidget(self.messages.count() - 1, note)
            toast(self.window(), "Chargement impossible.", "danger")

        loader.loaded.connect(on_loaded)
        loader.failed.connect(on_failed)
        loader.start()

    def _tokenizer_name(self) -> str:
        return {"char": "de caractères", "word": "de mots", "bpe": "BPE"}.get(
            self.project.tokenizer_kind, self.project.tokenizer_kind
        )

    # ------------------------------------------------------------------
    def _sampling_params(self) -> chat.SamplingParams:
        return chat.SamplingParams(
            temperature=self.temp_slider.value() / 100,
            top_k=self.topk_spin.value(),
            top_p=self.topp_slider.value() / 100,
            max_new_tokens=self.maxtok_spin.value(),
            repetition_penalty=self.rep_slider.value() / 100,
        )

    def _build_prompt(self, text: str) -> str:
        key = self.template_combo.currentData()
        spec = chat.PROMPT_TEMPLATES.get(key, {})
        template = spec.get("template", "{user}")
        if not self.history_toggle.isChecked():
            return chat.build_prompt(key, text)
        # historique borné pour rester dans le contexte du modèle
        transcript = ""
        for role, content in self.turns[-8:]:
            if role == "user":
                transcript += template.format(user=content) if key != "dialogue" else \
                    f"Utilisateur : {content}\n"
            else:
                transcript += ("Assistant" if key == "dialogue" else "### Réponse :") + \
                    f"\n{content}\n\n"
        transcript = transcript[-2000:]
        return transcript

    def send(self) -> None:
        if self.model is None or (self.worker is not None and self.worker.isRunning()):
            return
        text = self.input.toPlainText().strip()
        if not text:
            return
        self.input.clear()
        self._remove_note()
        self.turns.append(("user", text))
        self._add_bubble(text, "user")

        prompt = self._build_prompt(text)
        self.current_bubble = self._add_bubble("", "assistant")
        self.status_label.setText("Génération…")
        self.send_btn.setEnabled(False)
        self.stop_btn.setVisible(True)

        worker = GenerateWorker(
            self.model, self.tokenizer, prompt, self._sampling_params(),
            chat.stop_strings(self.template_combo.currentData()), self.device,
        )
        self.worker = worker

        def on_token(piece: str) -> None:
            if self.current_bubble is not None:
                self.current_bubble.append_text(piece)
                bar = self.scroll.verticalScrollBar()
                bar.setValue(bar.maximum())

        def on_done(full_text: str) -> None:
            self.turns.append(("assistant", full_text))
            if self.current_bubble is not None:
                self.current_bubble.set_text(full_text.strip() or "(réponse vide)")
                self.current_bubble.set_footer(
                    f"{len(full_text.split())} mots · gabarit : {self.template_combo.currentText()}"
                )
            self._finish_generation()

        def on_failed(message: str) -> None:
            if self.current_bubble is not None:
                self.current_bubble.set_text(f"(erreur : {message})")
            self._finish_generation()

        worker.token.connect(on_token)
        worker.finished_text.connect(on_done)
        worker.failed.connect(on_failed)
        worker.start()

    def _finish_generation(self) -> None:
        self.send_btn.setEnabled(True)
        self.stop_btn.setVisible(False)
        self.status_label.setText("Prêt")
        self.current_bubble = None
        self.input.setFocus()

    def stop_generation(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()

    def _add_bubble(self, text: str, role: str) -> Bubble:
        bubble = Bubble(text, role)
        self.messages.insertWidget(self.messages.count() - 1, bubble)
        bar = self.scroll.verticalScrollBar()
        QTimer.singleShot(30, lambda: bar.setValue(bar.maximum()))
        return bubble

    def _remove_note(self) -> None:
        for i in range(self.messages.count()):
            widget = self.messages.itemAt(i).widget()
            if isinstance(widget, Bubble) and widget.role == "note":
                widget.setParent(None)
                return

    def new_conversation(self) -> None:
        self.stop_generation()
        for i in reversed(range(self.messages.count())):
            widget = self.messages.itemAt(i).widget()
            if isinstance(widget, Bubble):
                widget.setParent(None)
        self.turns.clear()
        self._empty_state()

    def copy_last(self) -> None:
        if self.turns:
            QGuiApplication.clipboard().setText(self.turns[-1][1])
            toast(self.window(), "Réponse copiée.", "mint")
