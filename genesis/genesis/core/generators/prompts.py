"""Prompts de génération courts, variés et conçus pour éviter le méta-discours."""
from __future__ import annotations

CATEGORIES = {
    "récits": "récits littéraires originaux, personnages et lieux variés",
    "dialogues": "dialogues naturels entre personnages aux voix distinctes",
    "articles": "articles explicatifs factuels et structurés sur des sujets différents",
    "lettres": "lettres personnelles fictives, avec situations et tons variés",
    "descriptions": "descriptions sensorielles de lieux, objets et scènes différentes",
    "mélange": "un mélange équilibré de narration, dialogue, explication et description",
}


def build_prompt(category: str, language: str, topic: str, style: str,
                target_words: int, previous_topics: list[str] | None = None) -> str:
    kind = CATEGORIES.get(category, CATEGORIES["mélange"])
    avoid = ", ".join((previous_topics or [])[-12:])
    topic_line = f"Thème ou consigne : {topic.strip()}" if topic.strip() else (
        "Choisis un sujet distinct et original, de préférence éloigné des exemples précédents."
    )
    style_line = f"Style souhaité : {style.strip()}" if style.strip() else (
        "Style clair et naturel ; varie la syntaxe et le vocabulaire."
    )
    avoid_line = f"Évite de reprendre ces thèmes récents : {avoid}." if avoid else ""
    return (
        f"Écris un texte original en {language}. Il doit être un exemple de {kind}.\n"
        f"{topic_line}\n{style_line}\n{avoid_line}\n"
        f"Longueur visée : environ {target_words} mots.\n"
        "Ne présente pas la consigne, ne mentionne pas cette demande, ne donne pas de titre "
        "générique, ne prétends pas citer une source réelle et n'ajoute aucun commentaire "
        "après le texte. Produis uniquement le texte. Évite les listes de formulations répétées."
    )
