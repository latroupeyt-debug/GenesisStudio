"""Dimensionnement : combien de texte faut-il pour qu'un modèle apprenne la langue
au lieu d'apprendre le corpus par cœur ?

Heuristiques utilisées (volontairement simples et explicites) :

    ratio = nombre de tokens du corpus (tokens en doublon compris) / nombre de paramètres du modèle

    ratio < 1     : le modèle est plus gros que le corpus → il récite.
    1 – 20        : risque élevé ; le modèle a peu d'exemples divers.
    20 – 100      : plancher indicatif, surveiller la perte de validation.
    100 – 300     : objectif prudent pour un petit GPT entraîné sur un corpus maison.
    > 300         : volume plus confortable, sans garantie automatique.

Référence : les lois d'échelle (Chinchilla) ont trouvé un optimum calcul/données
proche de 20 tokens par paramètre pour des entraînements de très grande échelle. Ce
n'est PAS un seuil anti-mémorisation transposable tel quel à un petit GPT maison.
Les seuils ci-dessous sont des avertissements conservateurs, pas une garantie :
la diversité, la déduplication, la difficulté du texte et la courbe de validation
comptent autant. Ajouter des époques n'augmente PAS les données uniques — seulement
le temps passé à les répéter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .model import PRESETS_BY_KEY, ModelConfig, count_parameters, human_params

LEVELS = {
    "critique": {"tone": "danger", "label": "critique"},
    "risque": {"tone": "amber", "label": "risqué"},
    "correct": {"tone": "mint", "label": "correct"},
    "bon": {"tone": "mint", "label": "bon"},
    "excellent": {"tone": "mint", "label": "confortable"},
}

CHARS_PER_TOKEN_DEFAULT = 4.0  # ~4 caractères par token BPE en français
MIN_RATIO = 20.0               # plancher de sélection, pas garantie anti-mémorisation
GOOD_RATIO = 100.0             # repère prudent pour un corpus maison


@dataclass
class SizingAdvice:
    params: int = 0
    corpus_tokens: int = 0
    epochs: int = 1
    ratio_corpus: float = 0.0
    ratio_exposure: float = 0.0
    level: str = "correct"
    headline: str = ""
    detail: str = ""
    recommended_epochs: int = 1
    tokens_for_minimum: int = 0
    tokens_for_comfort: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def tone(self) -> str:
        return LEVELS.get(self.level, LEVELS["correct"])["tone"]

    @property
    def badge(self) -> str:
        return LEVELS.get(self.level, LEVELS["correct"])["label"]


def _fmt(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} Md".replace(".0", "")
    if n >= 1e6:
        return f"{n / 1e6:.1f} M".replace(".0", "")
    if n >= 1e3:
        return f"{n / 1e3:.0f} k"
    return f"{int(n)}"


def advise(params: int, corpus_tokens: int, epochs: int = 1,
           chars_per_token: float = CHARS_PER_TOKEN_DEFAULT) -> SizingAdvice:
    """Analyse le couple (taille du modèle, taille du corpus)."""
    params = max(1, int(params))
    corpus_tokens = max(0, int(corpus_tokens))
    epochs = max(1, int(epochs))
    advice = SizingAdvice(
        params=params,
        corpus_tokens=corpus_tokens,
        epochs=epochs,
        tokens_for_minimum=int(MIN_RATIO * params),
        tokens_for_comfort=int(GOOD_RATIO * params),
    )
    advice.ratio_corpus = corpus_tokens / params
    advice.ratio_exposure = (corpus_tokens * epochs) / params

    ratio = advice.ratio_corpus
    if ratio < 1:
        advice.level = "critique"
        advice.headline = "Le modèle est plus gros que votre corpus"
        advice.detail = (
            f"{_fmt(corpus_tokens)} tokens pour {human_params(params)} paramètres "
            f"({ratio:.1f} token/paramètre). Le modèle va apprendre le corpus par cœur : "
            "il recopiera des passages au lieu de produire du texte nouveau. "
            f"Ajoutez du texte (au moins {_fmt(advice.tokens_for_minimum)} tokens, "
            f"soit ~{advice.tokens_for_minimum * chars_per_token / 1e6:.0f} Mo de texte) "
            "ou choisissez un gabarit plus petit."
        )
        advice.recommended_epochs = 1
    elif ratio < MIN_RATIO:
        advice.level = "risque"
        advice.headline = "Corpus trop juste pour un apprentissage robuste"
        advice.detail = (
            f"{_fmt(corpus_tokens)} tokens pour {human_params(params)} paramètres "
            f"({ratio:.1f} tokens/paramètre). Risque élevé de répétition et de "
            "mémorisation. Le repère de 20 tokens/paramètre n'est qu'un plancher de "
            "sélection, pas une garantie. Ajoutez des textes variés ou choisissez un "
            f"gabarit plus petit (repère bas : {_fmt(advice.tokens_for_minimum)} tokens)."
        )
        advice.recommended_epochs = 1
    elif ratio < GOOD_RATIO:
        advice.level = "correct"
        advice.headline = "Plancher indicatif atteint — surveillez la validation"
        advice.detail = (
            f"{ratio:.0f} tokens par paramètre : volume de départ, mais pas de garantie "
            "contre le par-cœur, surtout si les textes se répètent. Le repère prudent est "
            f"{_fmt(advice.tokens_for_comfort)} tokens (~{advice.tokens_for_comfort * chars_per_token / 1e6:.0f} Mo). "
            "Surveillez la perte de validation et arrêtez si elle stagne ou remonte."
        )
        advice.recommended_epochs = 1
    elif ratio < 300:
        advice.level = "bon"
        advice.headline = "Volume prudent pour un corpus maison"
        advice.detail = (
            f"{ratio:.0f} tokens par paramètre : bon point de départ si le corpus est "
            "réellement varié et dédupliqué. Ce ratio ne garantit pas à lui seul la "
            "généralisation ; surveillez la validation. Une à deux époques suffisent."
        )
        advice.recommended_epochs = 1
    else:
        advice.level = "excellent"
        advice.headline = "Corpus très fourni pour ce gabarit"
        advice.detail = (
            f"{ratio:.0f} tokens par paramètre : volume confortable, sous réserve de "
            "variété et de qualité. Le calcul peut devenir le facteur limitant. Vous pouvez "
            "tester un gabarit plus grand ; la validation reste l'arbitre."
        )
        advice.recommended_epochs = 1

    # avertissement quand le même corpus est répété de nombreuses fois
    if epochs >= 3 and corpus_tokens > 0:
        advice.warnings.append(
            f"{epochs} époques réutilisent le même corpus {epochs} fois : cela ne crée pas "
            "de nouveaux exemples et peut favoriser le par-cœur. Commencez par une époque "
            "et surveillez la perte de validation."
        )
    if corpus_tokens and corpus_tokens < 20_000:
        advice.warnings.append(
            "Moins de 20 k tokens : le corpus est trop petit pour tirer une conclusion fiable. "
            "Ajoutez des textes variés."
        )
    return advice


def preset_table(vocab_size: int, chars_per_token: float = CHARS_PER_TOKEN_DEFAULT,
                 presets: list[str] | None = None) -> list[dict]:
    """Tableau « gabarit → corpus minimal / confortable », pour le guide."""
    rows = []
    keys = presets or [p["key"] for p in PRESETS_BY_KEY.values() if p["key"] != "custom"]
    for key in keys:
        preset = PRESETS_BY_KEY.get(key)
        if not preset:
            continue
        cfg = ModelConfig(
            vocab_size=max(2, vocab_size),
            block_size=int(preset["block_size"]),
            n_layer=int(preset["n_layer"]),
            n_head=int(preset["n_head"]),
            n_embd=int(preset["n_embd"]),
        )
        params = count_parameters(cfg)
        minimum = MIN_RATIO * params
        comfort = GOOD_RATIO * params
        rows.append({
            "key": key,
            "label": preset["label"],
            "params": params,
            "tokens_min": int(minimum),
            "tokens_comfort": int(comfort),
            "mb_min": minimum * chars_per_token / 1e6,
            "mb_comfort": comfort * chars_per_token / 1e6,
            "speed": preset["speed"],
        })
    return rows


def corpus_size_hint(tokens: int, chars_per_token: float = CHARS_PER_TOKEN_DEFAULT) -> str:
    chars = tokens * chars_per_token
    if chars >= 1e9:
        return f"~{chars / 1e9:.1f} Md de caractères"
    return f"~{chars / 1e6:.1f} Mo de texte".replace(".0 Mo", " Mo")
