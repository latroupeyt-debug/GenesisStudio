"""Composants graphiques réutilisables (aucune dépendance hors PySide6)."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import theme


# ---------------------------------------------------------------------------
# Libellés
# ---------------------------------------------------------------------------
def label(text: str = "", kind: str = "", wrap: bool = False) -> QLabel:
    lbl = QLabel(text)
    if kind:
        lbl.setObjectName(kind)
    lbl.setWordWrap(wrap)
    return lbl


def h1(text: str) -> QLabel:
    return label(text, "H1")


def h2(text: str) -> QLabel:
    return label(text, "H2")


def h3(text: str) -> QLabel:
    return label(text, "H3")


def muted(text: str = "", wrap: bool = True) -> QLabel:
    return label(text, "Muted", wrap)


def dim(text: str = "", wrap: bool = True) -> QLabel:
    return label(text, "Dim", wrap)


def separator() -> QFrame:
    line = QFrame()
    line.setObjectName("Separator")
    line.setFixedHeight(1)
    return line


# ---------------------------------------------------------------------------
# Conteneurs
# ---------------------------------------------------------------------------
class Card(QFrame):
    def __init__(self, parent: QWidget | None = None, soft: bool = False, margins: int = 20, spacing: int = 12):
        super().__init__(parent)
        self.setObjectName("CardSoft" if soft else "Card")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(margins, margins, margins, margins)
        self.body.setSpacing(spacing)


class ScrollArea(QScrollArea):
    def __init__(self, parent: QWidget | None = None, transparent: bool = True):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        if transparent:
            self.viewport().setAutoFillBackground(False)
        self.container = QWidget()
        self.container.setAutoFillBackground(False)
        self.setWidget(self.container)


class Chip(QLabel):
    TONES = {
        "neutral": ("#1B2130", theme.TEXT_MUTED, "#252D3D"),
        "accent": (theme.ACCENT_SOFT, "#C4BCFF", "#443C80"),
        "mint": ("rgba(46, 211, 168, 0.12)", "#6FE7C4", "#1E4A3C"),
        "amber": ("rgba(255, 180, 61, 0.12)", "#FFCB73", "#4A3A1A"),
        "danger": (theme.DANGER_SOFT, "#FF9AA2", "#4A2730"),
    }

    def __init__(self, text: str = "", tone: str = "neutral", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.set_tone(tone)

    def set_tone(self, tone: str) -> None:
        bg, fg, border = self.TONES.get(tone, self.TONES["neutral"])
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border: 1px solid {border};"
            "border-radius: 8px; padding: 3px 9px; font-size: 11px; font-weight: 600;"
        )


class IconButton(QPushButton):
    def __init__(self, icon_name: str, tooltip: str = "", size: int = 32,
                 color: str = theme.TEXT_MUTED, parent: QWidget | None = None,
                 ghost: bool = True):
        super().__init__(parent)
        self.setIcon(theme.svg_icon(icon_name, color, 17))
        self.setIconSize(QSize(17, 17))
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(tooltip)
        self.setObjectName("Ghost" if ghost else "")
        self.setStyleSheet("padding: 0px; border-radius: 8px;" if ghost else "")


class SegmentedControl(QWidget):
    changed = Signal(int)

    def __init__(self, items: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self._buttons: list[QPushButton] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.setStyleSheet(
            f"background: {theme.SURFACE}; border: 1px solid {theme.BORDER_SOFT}; border-radius: 12px;"
        )
        for i, text in enumerate(items):
            btn = QPushButton(text)
            btn.setObjectName("Segment")
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _=False, idx=i: self.set_index(idx))
            layout.addWidget(btn)
            self._buttons.append(btn)
        if self._buttons:
            self._buttons[0].setChecked(True)

    def set_index(self, index: int) -> None:
        for i, btn in enumerate(self._buttons):
            btn.setChecked(i == index)
        self.changed.emit(index)

    def current_index(self) -> int:
        for i, btn in enumerate(self._buttons):
            if btn.isChecked():
                return i
        return 0


class Collapsible(QWidget):
    """Section repliable avec un en-tête cliquable."""

    def __init__(self, title: str, content: QWidget, expanded: bool = False,
                 parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.header = QPushButton()
        self.header.setObjectName("Ghost")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet("text-align: left; padding: 6px 2px; font-weight: 600;")
        self.header.setLayoutDirection(Qt.RightToLeft)
        self.header.clicked.connect(self.toggle)
        layout.addWidget(self.header)

        self.content = content
        self.content.setVisible(expanded)
        layout.addWidget(self.content)
        self._expanded = expanded
        self._title = title
        self._refresh()

    def _refresh(self) -> None:
        arrow = "chevron-down" if self._expanded else "chevron-right"
        self.header.setIcon(theme.svg_icon(arrow, theme.TEXT_MUTED, 15))
        self.header.setText("  " + self._title)

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self.content.setVisible(self._expanded)
        self._refresh()


class StatTile(QFrame):
    """Petite tuile « valeur + libellé », façon tableau de bord."""

    def __init__(self, value: str, caption: str, accent: str | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("CardSoft")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(1)
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(
            f"font-size: 15.5px; font-weight: 700; color: {accent or theme.TEXT};"
        )
        self.caption_label = QLabel(caption)
        self.caption_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 10.5px;")
        layout.addWidget(self.value_label)
        layout.addWidget(self.caption_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


# ---------------------------------------------------------------------------
# Zone de glisser-déposer
# ---------------------------------------------------------------------------
class DropZone(QFrame):
    filesDropped = Signal(list)
    clicked = Signal()

    def __init__(self, title: str, subtitle: str, icon_name: str = "upload",
                 parent: QWidget | None = None, height: int = 168):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(height)
        self.title = title
        self.subtitle = subtitle
        self.icon_name = icon_name
        self._hover = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignCenter)

        self.icon_label = QLabel()
        self.icon_label.setPixmap(theme.svg_icon(icon_name, theme.ACCENT, 34).pixmap(34, 34))
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 15px; font-weight: 650;")
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setAlignment(Qt.AlignCenter)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 12px;")
        self.subtitle_label.setMaximumWidth(560)
        self.subtitle_label.setMinimumWidth(420)

        layout.addWidget(self.icon_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.subtitle_label)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1.5, 1.5, -1.5, -1.5)
        bg = theme.color(theme.ACCENT, 0.07) if self._hover else theme.color(theme.SURFACE_2, 0.55)
        painter.setBrush(bg)
        pen = QPen(theme.color(theme.ACCENT if self._hover else "#2E3646"))
        pen.setWidthF(1.6)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([6, 4])
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 16, 16)
        painter.end()

    def _set_hover(self, value: bool) -> None:
        self._hover = value
        self.update()

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._set_hover(True)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._set_hover(False)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._set_hover(False)
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
        event.acceptProposedAction()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and not getattr(self, "_locked", False):
            self.clicked.emit()


# ---------------------------------------------------------------------------
# Graphique de perte
# ---------------------------------------------------------------------------
class LossChart(QWidget):
    """Courbe de perte dessinée à la main (pas de dépendance QtCharts)."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.train: list[tuple[float, float]] = []
        self.eval: list[tuple[float, float]] = []
        self.empty_text = "La courbe de perte apparaîtra pendant l'entraînement."

    def set_history(self, history: list) -> None:
        train, ev = [], []
        for row in history or []:
            try:
                step, loss = float(row[0]), float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if loss == loss:  # pas de NaN
                train.append((step, loss))
            if len(row) > 2 and row[2] is not None:
                try:
                    ev.append((step, float(row[2])))
                except (TypeError, ValueError):
                    pass
        self.train = train
        self.eval = ev
        self.update()

    def clear(self) -> None:
        self.train = []
        self.eval = []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0, 0, -1, -1)

        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.color(theme.SURFACE_2, 0.6))
        painter.drawRoundedRect(rect, 12, 12)

        plot = rect.adjusted(48, 16, -14, -26)

        if len(self.train) < 2:
            painter.setPen(theme.color(theme.TEXT_DIM))
            painter.setFont(theme.base_font(9))
            painter.drawText(rect, Qt.AlignCenter, self.empty_text)
            painter.end()
            return

        xs = [s for s, _ in self.train] + [s for s, _ in self.eval]
        ys = [l for _, l in self.train] + [l for _, l in self.eval]
        x_min, x_max = min(xs), max(xs)
        if x_max <= x_min:
            x_max = x_min + 1
        y_min, y_max = min(ys), max(ys)
        pad = max(0.05, (y_max - y_min) * 0.15)
        y_min, y_max = max(0.0, y_min - pad), y_max + pad

        # grille + graduations
        painter.setFont(theme.base_font(8))
        grid_pen = QPen(theme.color(theme.BORDER_SOFT, 0.9))
        grid_pen.setWidthF(1.0)
        for i in range(5):
            y = plot.top() + plot.height() * i / 4
            painter.setPen(grid_pen)
            painter.drawLine(int(plot.left()), int(y), int(plot.right()), int(y))
            value = y_max - (y_max - y_min) * i / 4
            painter.setPen(theme.color(theme.TEXT_DIM))
            painter.drawText(QRectF(rect.left() + 4, y - 8, 40, 16),
                             Qt.AlignRight | Qt.AlignVCenter, f"{value:.2f}")

        def to_point(step: float, loss: float) -> QPoint:
            fx = (step - x_min) / (x_max - x_min)
            fy = (loss - y_min) / max(1e-9, (y_max - y_min))
            return QPoint(int(plot.left() + fx * plot.width()),
                          int(plot.bottom() - fy * plot.height()))

        # aire + courbe d'entraînement
        path = QPainterPath()
        first = to_point(*self.train[0])
        path.moveTo(first.x(), plot.bottom())
        path.lineTo(first.x(), first.y())
        for step, loss in self.train[1:]:
            pt = to_point(step, loss)
            path.lineTo(pt)
        last = to_point(*self.train[-1])
        path.lineTo(last.x(), plot.bottom())
        path.closeSubpath()

        gradient = QLinearGradient(0, plot.top(), 0, plot.bottom())
        gradient.setColorAt(0.0, theme.color(theme.ACCENT, 0.34))
        gradient.setColorAt(1.0, theme.color(theme.ACCENT, 0.01))
        painter.setPen(Qt.NoPen)
        painter.setBrush(gradient)
        painter.drawPath(path)

        curve = QPainterPath()
        p0 = to_point(*self.train[0])
        curve.moveTo(p0)
        for step, loss in self.train[1:]:
            curve.lineTo(to_point(step, loss))
        pen = QPen(theme.color("#9C8FFF"))
        pen.setWidthF(2.0)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(curve)

        # points d'évaluation
        if self.eval:
            pen = QPen(theme.color(theme.MINT))
            pen.setWidthF(1.6)
            painter.setPen(pen)
            ev_path = QPainterPath()
            p0 = to_point(*self.eval[0])
            ev_path.moveTo(p0)
            for step, loss in self.eval[1:]:
                ev_path.lineTo(to_point(step, loss))
            painter.drawPath(ev_path)
            painter.setBrush(theme.color(theme.MINT))
            painter.setPen(Qt.NoPen)
            for step, loss in self.eval[-40:]:
                pt = to_point(step, loss)
                painter.drawEllipse(QRectF(pt.x() - 2, pt.y() - 2, 4, 4))

        # axe X
        painter.setPen(theme.color(theme.TEXT_DIM))
        painter.setFont(theme.base_font(8))
        painter.drawText(QRectF(plot.left(), rect.bottom() - 20, plot.width(), 16),
                         Qt.AlignLeft | Qt.AlignVCenter, f"pas {int(x_min)}")
        painter.drawText(QRectF(plot.left(), rect.bottom() - 20, plot.width(), 16),
                         Qt.AlignRight | Qt.AlignVCenter, f"pas {int(x_max)}")
        painter.drawText(QRectF(plot.left(), rect.top() - 14, plot.width(), 14),
                         Qt.AlignLeft | Qt.AlignVCenter, "perte")
        painter.end()


# ---------------------------------------------------------------------------
# Notifications éphémères
# ---------------------------------------------------------------------------
class Toast(QLabel):
    def __init__(self, parent: QWidget, text: str, tone: str = "accent", duration: int = 4200):
        super().__init__(parent)
        bg, fg = {
            "accent": (theme.ACCENT, "#FFFFFF"),
            "mint": ("#12463A", "#8DF0CE"),
            "danger": ("#4A1F27", "#FFB3BA"),
            "neutral": (theme.SURFACE_3, theme.TEXT),
        }.get(tone, (theme.SURFACE_3, theme.TEXT))
        self.setText(text)
        self.setWordWrap(True)
        self.setStyleSheet(
            f"background: {bg}; color: {fg}; border-radius: 12px; padding: 12px 18px;"
            "font-weight: 600; font-size: 12px; border: 1px solid rgba(255,255,255,0.10);"
        )
        self.setMaximumWidth(460)
        self.adjustSize()
        self._effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self._effect)
        self._anim = QPropertyAnimation(self._effect, b"opacity", self)
        self._anim.setDuration(260)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.reposition()
        self.show()
        self.raise_()
        self._anim.start()
        QTimer.singleShot(duration, self._fade_out)

    def reposition(self) -> None:
        parent = self.parentWidget()
        if not parent:
            return
        self.adjustSize()
        x = max(16, parent.width() - self.width() - 28)
        self.move(x, 24)

    def _fade_out(self) -> None:
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.0)
        self._anim.finished.connect(self.deleteLater)
        self._anim.start()


def toast(parent: QWidget, text: str, tone: str = "accent") -> None:
    Toast(parent, text, tone)


# ---------------------------------------------------------------------------
# Divers
# ---------------------------------------------------------------------------
def elide(text: str, width: int, font: QFont | None = None) -> str:
    metrics = QFontMetrics(font or theme.base_font(10))
    return metrics.elidedText(text, Qt.ElideRight, width)


def spacer(height: int = 0, width: int = 0) -> QWidget:
    w = QWidget()
    w.setFixedSize(width, height)
    w.setAttribute(Qt.WA_TransparentForMouseEvents)
    return w


def row(*widgets, spacing: int = 8, margins: tuple[int, int, int, int] = (0, 0, 0, 0)) -> QWidget:
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(*margins)
    layout.setSpacing(spacing)
    for widget in widgets:
        if isinstance(widget, int):
            layout.addStretch(widget)
        elif widget is None:
            layout.addStretch(1)
        else:
            layout.addWidget(widget)
    return container


# ---------------------------------------------------------------------------
# Carte sélectionnable (gabarit de modèle, type de tokenizer…)
# ---------------------------------------------------------------------------
class OptionCard(QFrame):
    clicked = Signal()

    def __init__(self, title: str, subtitle: str = "", description: str = "",
                 badge: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("CardSoft")
        self.setCursor(Qt.PointingHandCursor)
        self._selected = False
        self._badge_text = badge
        self._description = description
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-size: 13px; font-weight: 650;")
        head.addWidget(self.title_label)
        head.addStretch(1)
        self.badge_label = QLabel(badge)
        self.badge_label.setVisible(bool(badge))
        self.badge_label.setStyleSheet(
            f"color: {theme.TEXT_DIM}; font-size: 10.5px; font-weight: 600;"
        )
        head.addWidget(self.badge_label)
        layout.addLayout(head)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px;")
        self.subtitle_label.setVisible(bool(subtitle))
        layout.addWidget(self.subtitle_label)

        self.desc_label = QLabel(description)
        self.desc_label.setWordWrap(True)
        self.desc_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
        self.desc_label.setVisible(False)
        layout.addWidget(self.desc_label)
        self._apply_style()

    def set_locked(self, locked: bool) -> None:
        """Carte visible mais non cliquable (architecture héritée)."""
        self._locked = locked
        self.setCursor(Qt.ForbiddenCursor if locked else Qt.PointingHandCursor)
        effect = self.graphicsEffect()
        if locked:
            if effect is None:
                effect = QGraphicsOpacityEffect(self)
                self.setGraphicsEffect(effect)
            effect.setOpacity(0.40)
        elif effect is not None:
            effect.setOpacity(1.0)

    def set_badge(self, text: str) -> None:
        self.badge_label.setText(text)
        self.badge_label.setVisible(bool(text))

    def set_description(self, text: str) -> None:
        self._description = text
        self.desc_label.setText(text)
        self.desc_label.setVisible(bool(text))

    def set_expanded(self, expanded: bool) -> None:
        """Affiche la description détaillée (utile pour l'option sélectionnée)."""
        self.desc_label.setText(self._description)
        self.desc_label.setVisible(expanded and bool(self._description))

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()

    def is_selected(self) -> bool:
        return self._selected

    def _apply_style(self) -> None:
        if self._selected:
            self.setStyleSheet(
                f"QFrame#CardSoft {{ background: {theme.ACCENT_SOFT};"
                f" border: 1px solid #4A4090; border-radius: 14px; }}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#CardSoft {{ background: {theme.SURFACE_2};"
                f" border: 1px solid {theme.BORDER_SOFT}; border-radius: 14px; }}"
                f"QFrame#CardSoft:hover {{ border-color: #333C50; }}"
            )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and not getattr(self, "_locked", False):
            self.clicked.emit()


class FieldRow(QWidget):
    """Ligne « libellé à gauche, contrôle à droite », avec une aide dessous."""

    def __init__(self, title: str, widget: QWidget, hint: str = "",
                 parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        top = QHBoxLayout()
        top.setSpacing(10)
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px; font-weight: 500;")
        top.addWidget(title_label)
        top.addStretch(1)
        widget.setFixedWidth(130)
        top.addWidget(widget)
        layout.addLayout(top)
        if hint:
            hint_label = QLabel(hint)
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet(f"color: {theme.TEXT_DIM}; font-size: 11px;")
            layout.addWidget(hint_label)


class Sparkline(QWidget):
    """Mini-courbe de perte pour les cartes de version."""

    def __init__(self, points: list[list] | None = None, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(120, 44)
        self.setMaximumHeight(52)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.set_points(points or [])

    def set_points(self, history: list) -> None:
        self.points: list[tuple[float, float]] = []
        for row in history or []:
            try:
                if row[1] is None:
                    continue
                self.points.append((float(row[0]), float(row[1])))
            except (TypeError, ValueError, IndexError):
                continue
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        if len(self.points) < 2:
            painter.setPen(theme.color(theme.TEXT_DIM))
            painter.setFont(theme.base_font(8))
            painter.drawText(rect, Qt.AlignCenter, "pas encore de courbe")
            painter.end()
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        if x1 <= x0:
            x1 = x0 + 1
        if y1 - y0 < 1e-6:
            y1 = y0 + 1e-6

        def pt(step: float, loss: float) -> QPoint:
            fx = (step - x0) / (x1 - x0)
            fy = (loss - y0) / (y1 - y0)
            pad = 4
            return QPoint(int(rect.left() + pad + fx * (rect.width() - 2 * pad)),
                          int(rect.bottom() - pad - fy * (rect.height() - 2 * pad)))

        curve = QPainterPath()
        curve.moveTo(pt(*self.points[0]))
        for point in self.points[1:]:
            curve.lineTo(pt(*point))
        pen = QPen(theme.color("#8F82FF"))
        pen.setWidthF(1.6)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(curve)
        last = pt(*self.points[-1])
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.color(theme.MINT))
        painter.drawEllipse(QRectF(last.x() - 2.5, last.y() - 2.5, 5, 5))
        painter.end()
