"""Filtres locaux légers pour éviter d'ajouter des réponses vides ou dégradées."""
from __future__ import annotations

import re
import unicodedata

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def assess_text(text: str, min_chars: int = 120) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    stripped = text.strip()
    if len(stripped) < min_chars:
        warnings.append(f"texte trop court ({len(stripped)} caractères)")
    words = _WORD_RE.findall(stripped)
    if len(words) < 20:
        warnings.append("moins de 20 mots")
    letters = sum(ch.isalpha() for ch in stripped)
    visible = sum(not ch.isspace() for ch in stripped)
    if visible and letters / visible < 0.45:
        warnings.append("peu de lettres (format ou sortie possiblement dégradée)")
    lines = [normalize(line) for line in stripped.splitlines() if len(line.strip()) > 15]
    if lines and len(set(lines)) < len(lines) * 0.7:
        warnings.append("lignes fortement répétées")
    return not warnings, warnings
