"""Génération de corpus (API OpenAI-compatible ou Ollama local).

Les intégrations distantes sont facultatives ; aucun corpus existant n'est envoyé.
"""

from .client import Provider, GenerationError, generate_text, provider_choices
from .service import GenerationConfig, GenerationResult, generate_corpus

__all__ = [
    "Provider", "GenerationError", "generate_text", "provider_choices",
    "GenerationConfig", "GenerationResult", "generate_corpus",
]
