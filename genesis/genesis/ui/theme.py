"""Thème sombre de Genesis Studio : palette, feuille de style, icônes vectorielles."""

from __future__ import annotations

import sys

from PySide6.QtCore import QByteArray, QSize, Qt
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
BG = "#0A0C11"
SIDEBAR = "#0C0F15"
SURFACE = "#12161F"
SURFACE_2 = "#171C27"
SURFACE_3 = "#1D2431"
BORDER = "#232A38"
BORDER_SOFT = "#1B212C"
TEXT = "#E9EDF6"
TEXT_MUTED = "#8B94A9"
TEXT_DIM = "#5F6A80"
ACCENT = "#7C6CFF"
ACCENT_HOVER = "#8F82FF"
ACCENT_SOFT = "rgba(124, 108, 255, 0.16)"
ACCENT_SOFT_SOLID = "#2A2547"
MINT = "#2ED3A8"
AMBER = "#FFB43D"
DANGER = "#FF5F6D"
DANGER_SOFT = "rgba(255, 95, 109, 0.14)"

FONT_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans", "DejaVu Sans", "sans-serif"]
MONO_FAMILIES = ["Cascadia Mono", "Cascadia Code", "Consolas", "JetBrains Mono", "DejaVu Sans Mono", "monospace"]


def base_font(size: int = 10, weight: int = 400) -> QFont:
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSize(size)
    font.setWeight(QFont.Weight(weight))
    return font


def mono_font(size: int = 9) -> QFont:
    font = QFont()
    font.setFamilies(MONO_FAMILIES)
    font.setPointSize(size)
    return font


def dark_palette() -> QPalette:
    """Palette sombre : couvre tous les widgets non stylés par la feuille de style."""
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor("#0E121A"))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE_2))
    palette.setColor(QPalette.ToolTipBase, QColor(SURFACE_3))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(SURFACE_2))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Link, QColor("#A79BFF"))
    palette.setColor(QPalette.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    for group in (QPalette.Disabled,):
        palette.setColor(group, QPalette.Text, QColor(TEXT_DIM))
        palette.setColor(group, QPalette.WindowText, QColor(TEXT_DIM))
        palette.setColor(group, QPalette.ButtonText, QColor(TEXT_DIM))
    return palette


# ---------------------------------------------------------------------------
# Feuille de style globale
# ---------------------------------------------------------------------------
STYLESHEET = f"""
* {{
    outline: none;
}}
QWidget {{
    color: {TEXT};
    font-family: "Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans", sans-serif;
    font-size: 13px;
}}
QMainWindow, QDialog {{
    background: {BG};
}}
QToolTip {{
    background: {SURFACE_3};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 9px;
}}

/* ---------------- Cadres ---------------- */
QFrame#Sidebar {{
    background: {SIDEBAR};
    border-right: 1px solid {BORDER_SOFT};
}}
QFrame#Card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 16px;
}}
QFrame#CardSoft {{
    background: {SURFACE_2};
    border: 1px solid {BORDER_SOFT};
    border-radius: 14px;
}}
QFrame#Separator {{
    background: {BORDER_SOFT};
    max-height: 1px;
    min-height: 1px;
    border: none;
}}
QFrame#TopBar {{
    background: transparent;
}}

/* ---------------- Boutons ---------------- */
QPushButton {{
    background: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 9px 16px;
    font-weight: 500;
}}
QPushButton:hover {{
    background: {SURFACE_3};
    border-color: #2C3547;
}}
QPushButton:pressed {{
    background: #232B3B;
}}
QPushButton:disabled {{
    color: {TEXT_DIM};
    background: #14181F;
    border-color: {BORDER_SOFT};
}}
QPushButton#Primary {{
    background: {ACCENT};
    color: #FFFFFF;
    border: 1px solid {ACCENT_HOVER};
    font-weight: 600;
}}
QPushButton#Primary:hover {{
    background: {ACCENT_HOVER};
}}
QPushButton#Primary:pressed {{
    background: #6A5AE0;
}}
QPushButton#Primary:disabled {{
    background: #2A2F45;
    border-color: #333A52;
    color: #7E88A8;
}}
QPushButton#Ghost {{
    background: transparent;
    border: none;
    color: {TEXT_MUTED};
    padding: 7px 10px;
}}
QPushButton#Ghost:hover {{
    background: {SURFACE_2};
    color: {TEXT};
}}
QPushButton#Danger {{
    background: {DANGER_SOFT};
    border: 1px solid #4A2730;
    color: #FF8B94;
    font-weight: 600;
}}
QPushButton#Danger:hover {{
    background: rgba(255, 95, 109, 0.22);
    border-color: #7A3540;
}}
QPushButton#Chip {{
    background: {SURFACE_2};
    border: 1px solid {BORDER_SOFT};
    border-radius: 9px;
    padding: 6px 12px;
    color: {TEXT_MUTED};
    font-size: 12px;
    font-weight: 500;
}}
QPushButton#Chip:hover {{
    background: {SURFACE_3};
    color: {TEXT};
}}
QPushButton#Chip:checked {{
    background: {ACCENT_SOFT};
    border-color: #463C87;
    color: #C9C2FF;
}}

/* ---------------- Champs ---------------- */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: #0E121A;
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 9px 12px;
    color: {TEXT};
    selection-background-color: {ACCENT};
    selection-color: #FFFFFF;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: #4C4694;
    background: #0F1320;
}}
QLineEdit:disabled, QPlainTextEdit:disabled, QSpinBox:disabled, QComboBox:disabled {{
    color: {TEXT_DIM};
    background: #10141B;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
    margin-right: 4px;
}}
QComboBox::down-arrow {{
    image: url(__ARROW_DOWN__);
    width: 11px;
    height: 11px;
    margin-right: 7px;
}}
QComboBox QAbstractItemView {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 6px;
    selection-background-color: {ACCENT_SOFT};
    selection-color: {TEXT};
    outline: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    width: 17px;
    background: transparent;
    border: none;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    image: url(__ARROW_UP_SMALL__);
    width: 9px; height: 9px;
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    image: url(__ARROW_DOWN_SMALL__);
    width: 9px; height: 9px;
}}

/* ---------------- Sliders ---------------- */
QSlider::groove:horizontal {{
    height: 5px;
    background: #1E2532;
    border-radius: 3px;
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT};
    border-radius: 3px;
}}
QSlider::handle:horizontal {{
    background: #FFFFFF;
    width: 15px;
    height: 15px;
    margin: -6px 0;
    border-radius: 8px;
}}
QSlider::handle:horizontal:hover {{
    background: #E6E2FF;
}}

/* ---------------- Barre de progression ---------------- */
QProgressBar {{
    background: #1A202C;
    border: none;
    border-radius: 5px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 5px;
}}
QProgressBar#Mint::chunk {{
    background: {MINT};
}}

/* ---------------- Cases ---------------- */
QCheckBox {{
    spacing: 9px;
    color: {TEXT_MUTED};
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border-radius: 5px;
    border: 1px solid #2E3648;
    background: #0E121A;
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT_HOVER};
    image: url(__CHECK__);
}}
QCheckBox::indicator:hover {{
    border-color: #4A5470;
}}

/* ---------------- Onglets segmentés ---------------- */
QPushButton#Segment {{
    background: transparent;
    border: none;
    border-radius: 9px;
    padding: 8px 18px;
    color: {TEXT_MUTED};
    font-weight: 600;
}}
QPushButton#Segment:hover {{
    color: {TEXT};
}}
QPushButton#Segment:checked {{
    background: {SURFACE_3};
    color: {TEXT};
}}

/* ---------------- Barres de défilement ---------------- */
QScrollArea, QScrollArea > QWidget > QWidget, QAbstractScrollArea::viewport {{
    background: transparent;
}}
QScrollArea {{
    border: none;
}}
QStackedWidget {{
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #2A3141;
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: #39435A;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #2A3141;
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    background: none;
    width: 0;
}}

/* ---------------- Divers ---------------- */
QLabel#H1 {{ font-size: 26px; font-weight: 700; color: {TEXT}; }}
QLabel#H2 {{ font-size: 19px; font-weight: 650; color: {TEXT}; }}
QLabel#H3 {{ font-size: 15px; font-weight: 600; color: {TEXT}; }}
QLabel#Muted {{ color: {TEXT_MUTED}; }}
QLabel#Dim {{ color: {TEXT_DIM}; font-size: 12px; }}
QLabel#Mono {{ font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace; font-size: 12px; color: {TEXT_MUTED}; }}
QLabel#Danger {{ color: #FF8B94; }}
QLabel#Accent {{ color: #B3A9FF; }}
QMenu {{
    background: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 6px;
}}
QMenu::item {{
    padding: 8px 26px 8px 12px;
    border-radius: 7px;
}}
QMenu::item:selected {{
    background: {SURFACE_3};
}}
QMenu::separator {{
    height: 1px;
    background: {BORDER};
    margin: 5px 8px;
}}
QListWidget {{
    background: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    border-radius: 10px;
    padding: 8px;
    color: {TEXT_MUTED};
}}
QListWidget::item:selected {{
    background: {SURFACE_2};
    color: {TEXT};
}}
QListWidget::item:hover {{
    background: {SURFACE_2};
}}
"""


# ---------------------------------------------------------------------------
# Icônes SVG (rendues à la volée, donc nettes en haute densité)
# ---------------------------------------------------------------------------
_ICONS: dict[str, str] = {
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "trash": '<path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2m-9 0 1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13"/>',
    "chat": '<path d="M21 12a8 8 0 0 1-8 8H8l-4 3v-5.5A8 8 0 0 1 13 4a8 8 0 0 1 8 8Z"/>',
    "play": '<path d="M7 4.5v15l13-7.5-13-7.5Z"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2"/>',
    "folder": '<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
    "upload": '<path d="M12 16V4m0 0L7 9m5-5 5 5M4 17v2a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-2"/>',
    "chevron-left": '<path d="M15 5l-7 7 7 7"/>',
    "chevron-right": '<path d="M9 5l7 7-7 7"/>',
    "chevron-down": '<path d="M5 9l7 7 7-7"/>',
    "check": '<path d="M5 13l4 4L19 7"/>',
    "spark": '<path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4L12 3Z"/>',
    "gear": '<circle cx="12" cy="12" r="3.2"/><path d="M12 3v2.2M12 18.8V21M3 12h2.2M18.8 12H21M5.6 5.6l1.6 1.6M16.8 16.8l1.6 1.6M18.4 5.6l-1.6 1.6M7.2 16.8l-1.6 1.6"/>',
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 6M20 5v6h-6"/>',
    "file": '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5"/>',
    "info": '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.6v.2"/>',
    "warn": '<path d="M12 4l9 16H3l9-16Z"/><path d="M12 10v4M12 17v.2"/>',
    "send": '<path d="M4 12l16-8-6 16-3.4-6.6L4 12Z"/>',
    "chart": '<path d="M4 20V4M4 20h16"/><path d="M7 16l4-5 3 3 4-6"/>',
    "bolt": '<path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5Z"/><path d="M3 13l9 5 9-5"/>',
    "download": '<path d="M12 4v12m0 0 5-5m-5 5-5-5M4 20h16"/>',
    "copy": '<rect x="9" y="9" width="11" height="11" rx="2"/><path d="M15 5.5A1.5 1.5 0 0 0 13.5 4H6a2 2 0 0 0-2 2v7.5A1.5 1.5 0 0 0 5.5 15"/>',
    "close": '<path d="M6 6l12 12M18 6 6 18"/>',
    "database": '<ellipse cx="12" cy="6" rx="7" ry="3"/><path d="M5 6v12c0 1.7 3.1 3 7 3s7-1.3 7-3V6"/><path d="M5 12c0 1.7 3.1 3 7 3s7-1.3 7-3"/>',
    "dots": '<path d="M6 12h.01M12 12h.01M18 12h.01" stroke-width="2.6"/>',
    "brain": '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-1 5.8V15a3 3 0 0 0 3 3h1V4H9Z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 1 5.8V15a3 3 0 0 1-3 3h-1V4h0Z"/>',
}


def svg_icon(name: str, color: str = TEXT_MUTED, size: int = 18, stroke: float = 1.7) -> QIcon:
    body = _ICONS.get(name, "")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{color}" stroke-width="{stroke}" stroke-linecap="round" '
        f'stroke-linejoin="round">{body}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    scale = 2
    pm = QPixmap(QSize(size * scale, size * scale))
    pm.fill(Qt.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(scale)
    return QIcon(pm)


def color(hex_str: str, alpha: float | None = None) -> QColor:
    c = QColor(hex_str)
    if alpha is not None:
        c.setAlphaF(alpha)
    return c


def apply_dark_titlebar(widget) -> None:
    """Barre de titre sombre sur Windows 10/11 (sans effet ailleurs)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = int(widget.winId())
        for attribute in (20, 19):
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            )
    except Exception:
        pass


def app_icon() -> QIcon:
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        f'<rect x="2" y="2" width="28" height="28" rx="9" fill="{ACCENT}"/>'
        '<path d="M16 8l2.4 5.6L24 16l-5.6 2.4L16 24l-2.4-5.6L8 16l5.6-2.4L16 8Z" fill="#FFFFFF"/>'
        "</svg>"
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        pm = QPixmap(QSize(size, size))
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing)
        renderer.render(painter)
        painter.end()
        icon.addPixmap(pm)
    return icon


# ---------------------------------------------------------------------------
# Petites icônes utilisées *dans* la feuille de style (flèches, coche)
# ---------------------------------------------------------------------------
_STYLE_ICONS = {
    "arrow-down": ('<path d="M6 9l6 6 6-6"/>', TEXT_MUTED, 2.4, 20),
    "arrow-down-small": ('<path d="M6 9l6 6 6-6"/>', "#7A839A", 3.0, 16),
    "arrow-up-small": ('<path d="M6 15l6-6 6 6"/>', "#7A839A", 3.0, 16),
    "check": ('<path d="M5 13l4 4L19 7"/>', "#FFFFFF", 3.0, 18),
}


def _style_icon_path(name: str) -> str:
    from ..core import paths

    path_data, stroke, width, size = _STYLE_ICONS[name]
    cache = paths.home() / "cache" / "ui"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{name}.svg"
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="{size}" '
        f'height="{size}" fill="none" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round">{path_data}</svg>'
    )
    if not target.exists() or target.read_text(encoding="utf-8") != svg:
        target.write_text(svg, encoding="utf-8")
    return target.as_posix()


def _finalize_stylesheet() -> str:
    sheet = STYLESHEET
    for key, placeholder in (
        ("arrow-down", "__ARROW_DOWN__"),
        ("arrow-down-small", "__ARROW_DOWN_SMALL__"),
        ("arrow-up-small", "__ARROW_UP_SMALL__"),
        ("check", "__CHECK__"),
    ):
        try:
            sheet = sheet.replace(placeholder, _style_icon_path(key))
        except Exception:  # disque en lecture seule : on garde le placeholder
            sheet = sheet.replace(placeholder, "none")
    return sheet


STYLESHEET = _finalize_stylesheet()
