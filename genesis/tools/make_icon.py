"""Génère assets/genesis.ico et assets/genesis.png (icône de l'application).

    python tools/make_icon.py

Nécessite Pillow (uniquement pour la génération de l'icône, pas au runtime).
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
ACCENT = (124, 108, 255, 255)
ACCENT_TOP = (150, 136, 255, 255)
WHITE = (255, 255, 255, 255)


def render(size: int) -> Image.Image:
    scale = 4
    s = size * scale
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # fond dégradé arrondi
    radius = int(s * 0.28)
    for y in range(s):
        t = y / max(1, s - 1)
        color = tuple(int(ACCENT_TOP[i] + (ACCENT[i] - ACCENT_TOP[i]) * t) for i in range(3)) + (255,)
        draw.line([(0, y), (s, y)], fill=color)
    mask = Image.new("L", (s, s), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, s - 1, s - 1], radius=radius, fill=255)
    img.putalpha(mask)

    # étoile à quatre branches centrée
    cx = cy = s / 2
    outer = s * 0.34
    inner = s * 0.115
    points = []
    for i in range(8):
        angle = math.pi / 4 * i - math.pi / 2
        r = outer if i % 2 == 0 else inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=WHITE)

    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sizes = [16, 24, 32, 48, 64, 128, 256]
    images = [render(size) for size in sizes]
    images[-1].save(ASSETS / "genesis.ico", format="ICO", sizes=[(s, s) for s in sizes])
    images[-1].save(ASSETS / "genesis.png")
    print("Icônes générées :", ASSETS / "genesis.ico", "et", ASSETS / "genesis.png")


if __name__ == "__main__":
    main()
