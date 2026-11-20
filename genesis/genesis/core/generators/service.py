"""Orchestration of corpus generation with bounded requests and partial-file recovery."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .client import GenerationError, generate_text
from .dedup import Deduplicator
from .prompts import build_prompt
from .quality import assess_text


@dataclass
class GenerationConfig:
    provider: str = "ollama"
    model: str = "qwen2.5:3b"
    api_key: str = ""  # never serialized to a project/settings file
    category: str = "mélange"
    language: str = "français"
    topic: str = ""
    style: str = ""
    target_tokens: int = 10_000
    max_requests: int = 20
    max_tokens_per_request: int = 1024
    temperature: float = 0.8


@dataclass
class GenerationResult:
    path: Path | None = None
    chars: int = 0
    approx_tokens: int = 0
    accepted: int = 0
    duplicates: int = 0
    low_quality: int = 0
    requests: int = 0
    target_tokens: int = 0
    request_limit: int = 0
    stopped: bool = False
    error: str = ""
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.stopped:
            status = "arrêt partiel"
        elif self.error:
            status = "interrompu"
        elif self.request_limit and self.requests >= self.request_limit and self.approx_tokens < self.target_tokens:
            status = "limite de requêtes atteinte"
        else:
            status = "terminé"
        return (
            f"{status} · {self.accepted} texte(s) · ~{self.approx_tokens:,} tokens · "
            f"{self.requests} requête(s)".replace(",", " ")
        )


def _unique_path(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = folder / f"generation_ia_{stamp}.txt"
    n = 2
    while candidate.exists():
        candidate = folder / f"generation_ia_{stamp}_{n}.txt"
        n += 1
    return candidate


def generate_corpus(config: GenerationConfig, output_dir: Path, *,
                    progress: Callable[[float, str], None] | None = None,
                    should_stop: Callable[[], bool] | None = None) -> GenerationResult:
    """Génère de petits documents indépendants ; ne lit/expédie jamais le corpus existant.

    Le volume est approximé à 4 caractères/token. Même si un quota ou le réseau échoue,
    les textes déjà produits sont gardés dans un fichier partiel, importable dans l'app.
    """
    target = max(100, min(5_000_000, int(config.target_tokens)))
    max_requests = max(1, min(1000, int(config.max_requests)))
    max_tokens = max(64, min(4096, int(config.max_tokens_per_request)))
    result = GenerationResult(target_tokens=target, request_limit=max_requests)
    output = _unique_path(Path(output_dir))
    temp = output.with_suffix(".part")
    dedup = Deduplicator()
    topics: list[str] = []
    char_count = 0

    try:
        with temp.open("w", encoding="utf-8", newline="\n") as stream:
            while result.requests < max_requests:
                if should_stop and should_stop():
                    result.stopped = True
                    break
                if result.approx_tokens >= target:
                    break
                remaining = max(0, target - result.approx_tokens)
                words = max(80, min(1600, int(min(max_tokens * 0.65, remaining * 0.7))))
                prompt = build_prompt(
                    config.category, config.language, config.topic, config.style,
                    words, topics,
                )

                def count_attempt() -> bool:
                    if result.requests >= max_requests:
                        return False
                    result.requests += 1
                    if progress:
                        progress(min(0.95, result.approx_tokens / max(1, target)),
                                 f"Requête {result.requests}/{max_requests} · environ "
                                 f"{result.approx_tokens:,}/{target:,} tokens".replace(",", " "))
                    return True

                try:
                    text = generate_text(
                        config.provider, config.model, prompt, api_key=config.api_key,
                        max_tokens=max_tokens, temperature=config.temperature,
                        should_stop=should_stop, on_attempt=count_attempt,
                    )
                except GenerationError as exc:
                    if "annulée" in str(exc).casefold():
                        result.stopped = True
                    else:
                        result.error = str(exc)
                    break

                ok, warnings = assess_text(text)
                if not ok:
                    result.low_quality += 1
                    result.warnings.extend(warnings[:2])
                    if progress:
                        progress(min(0.98, result.approx_tokens / max(1, target)),
                                 f"Texte ignoré par le filtre qualité : {', '.join(warnings[:2])}")
                    continue
                if not dedup.add(text):
                    result.duplicates += 1
                    if progress:
                        progress(min(0.98, result.approx_tokens / max(1, target)),
                                 "Texte trop similaire à une réponse précédente : ignoré.")
                    continue

                clean = text.strip()
                stream.write(clean + "\n\n")
                stream.flush()
                char_count += len(clean) + 2
                result.accepted += 1
                result.approx_tokens = char_count // 4
                topic_hint = (config.topic.strip() or clean.splitlines()[0])[:80]
                topics.append(topic_hint)
                if progress:
                    progress(min(0.99, result.approx_tokens / max(1, target)),
                             f"Texte {result.accepted} accepté · ~{result.approx_tokens:,} tokens".replace(",", " "))

        if char_count:
            temp.replace(output)
            result.path = output
            result.chars = char_count
        else:
            temp.unlink(missing_ok=True)
        if result.requests >= max_requests and result.approx_tokens < target and not result.error and not result.stopped:
            result.warnings.append("Le plafond de requêtes choisi a été atteint avant le volume visé.")
        if result.accepted == 0 and not result.error and not result.stopped:
            result.error = "Aucun texte exploitable n'a été produit. Essayez un autre modèle ou une consigne différente."
    except Exception as exc:
        result.error = str(exc) or type(exc).__name__
        if temp.exists() and temp.stat().st_size:
            temp.replace(output)
            result.path = output
            result.chars = output.stat().st_size
            result.approx_tokens = result.chars // 4
        else:
            temp.unlink(missing_ok=True)
    if progress:
        progress(1.0, result.summary())
    return result
