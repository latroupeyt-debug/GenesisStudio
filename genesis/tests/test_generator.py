"""Tests locaux du générateur : prompts, déduplication, filtres et sortie partielle."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis.core.generators.client import GenerationError, generate_text  # noqa: E402
import genesis.core.generators.client as client_mod
from genesis.core.generators.dedup import Deduplicator, similarity  # noqa: E402
from genesis.core.generators.prompts import build_prompt  # noqa: E402
from genesis.core.generators.quality import assess_text  # noqa: E402
from genesis.core.generators.service import GenerationConfig, generate_corpus  # noqa: E402


SAMPLE_A = (
    "Au lever du jour, la place du village s'éveillait doucement. Les volets bleus "
    "s'ouvraient un à un tandis que le boulanger déposait des pains encore chauds. "
    "Une brise légère traversait les platanes et faisait danser les feuilles. "
    "Sur le banc près de la fontaine, deux voisins échangeaient des nouvelles et riaient. "
    "Le marché allait bientôt commencer dans les ruelles ensoleillées du vieux quartier."
)
SAMPLE_B = (
    "La nuit tombait sur la montagne lorsque Noémie aperçut une lueur derrière les pins. "
    "Elle posa son sac, écouta le vent et suivit le sentier jusqu'à une petite cabane. "
    "À l'intérieur, une lampe éclairait une carte ancienne couverte de signes. "
    "Un carnet racontait le voyage d'une famille venue observer les étoiles. "
    "Noémie sourit : elle avait trouvé le refuge idéal pour attendre l'aube."
)


def main() -> int:
    prompt = build_prompt("récits", "français", "une vallée", "sobre", 180, ["la mer"])
    assert "une vallée" in prompt and "la mer" in prompt and "180 mots" in prompt
    ok, warnings = assess_text(SAMPLE_A)
    assert ok, warnings
    assert not assess_text("réponse vide")[0]
    assert similarity(SAMPLE_A, SAMPLE_A) == 1.0

    dedup = Deduplicator()
    assert dedup.add(SAMPLE_A)
    assert not dedup.add("  " + SAMPLE_A.lower().replace("\n", " ") + "  ")
    assert dedup.add(SAMPLE_B)

    # Les appels payants ou sans clés sont bloqués avant tout accès réseau.
    try:
        generate_text("openrouter", "some/provider", "prompt", api_key="fake")
        raise AssertionError("un modèle OpenRouter non gratuit doit être refusé")
    except GenerationError as exc:
        assert "se terminant" in str(exc)
    try:
        generate_text("groq", "openai/gpt-oss-120b", "prompt")
        raise AssertionError("une clé vide doit être refusée")
    except GenerationError as exc:
        assert "clé API" in str(exc)

    # Les appels Ollama ont leur payload local dédié ; aucune clé API n'est nécessaire.
    captured = {}
    def fake_post(url, headers, body, timeout=90):
        captured.update(url=url, headers=headers, body=body)
        return {"message": {"content": SAMPLE_A}}, {}
    with patch("genesis.core.generators.client._post_json", side_effect=fake_post):
        local_text = generate_text("ollama", "qwen2.5:3b", prompt, on_attempt=lambda: True)
    assert local_text == SAMPLE_A
    assert captured["url"].startswith("http://127.0.0.1")
    assert captured["body"]["stream"] is False
    assert "Authorization" not in captured["headers"]

    attempts = []
    def fake_retry(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise GenerationError("429 retry", retryable=True)
        return {"choices": [{"message": {"content": SAMPLE_A}}]}, {}
    with patch("genesis.core.generators.client._post_json", side_effect=fake_retry), \
         patch("genesis.core.generators.client.time.sleep", return_value=None):
        result_text = generate_text("groq", "model", prompt, api_key="test", on_attempt=lambda: True)
    assert result_text == SAMPLE_A and len(attempts) == 2

    # Un texte répété est filtré ; le second texte distinct est conservé dans un fichier.
    answers = iter([SAMPLE_A, SAMPLE_A, SAMPLE_B])
    with tempfile.TemporaryDirectory(prefix="genesis-generator-") as temp:
        config = GenerationConfig(
            provider="ollama", model="test-local", target_tokens=1000,
            max_requests=3, max_tokens_per_request=512,
        )
        def fake_generate(*args, **kwargs):
            assert kwargs["on_attempt"]()
            return next(answers)
        with patch("genesis.core.generators.service.generate_text", side_effect=fake_generate):
            result = generate_corpus(config, Path(temp))
        assert result.path and result.path.exists()
        assert result.accepted == 2 and result.duplicates == 1
        output = result.path.read_text(encoding="utf-8")
        assert SAMPLE_A in output and SAMPLE_B in output
        print("service:", result.summary())

    # Un dépassement de quota après un texte garde la sortie partielle.
    with tempfile.TemporaryDirectory(prefix="genesis-generator-partial-") as temp:
        config = GenerationConfig(provider="ollama", model="test", target_tokens=5000,
                                  max_requests=3, max_tokens_per_request=512)
        answers = iter([SAMPLE_A, GenerationError("quota 429")])
        def fake_partial(*args, **kwargs):
            assert kwargs["on_attempt"]()
            item = next(answers)
            if isinstance(item, Exception):
                raise item
            return item
        with patch("genesis.core.generators.service.generate_text", side_effect=fake_partial):
            result = generate_corpus(config, Path(temp))
        assert result.path and result.path.exists() and result.error == "quota 429"
        assert result.requests == 2
        assert SAMPLE_A in result.path.read_text(encoding="utf-8")
        print("partiel conservé:", result.path.name, "·", result.error)

    print("TOUS LES TESTS DU GÉNÉRATEUR SONT PASSÉS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
