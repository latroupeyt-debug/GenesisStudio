"""Clients API pour Groq, OpenRouter et Ollama local (stdlib seulement).

Gemini est volontairement absent : ses conditions actuelles excluent la génération
pour développer des modèles concurrents, et le quota gratuit n'est pas adapté à cet usage.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Provider:
    key: str
    label: str
    base_url: str
    default_model: str
    needs_key: bool
    note: str


PROVIDERS = {
    "ollama": Provider(
        "ollama", "Ollama (local, privé)", "http://127.0.0.1:11434/api/chat",
        "qwen2.5:3b", False,
        "Aucun texte ni prompt ne quitte votre ordinateur. Ollama doit être installé et lancé.",
    ),
    "groq": Provider(
        "groq", "Groq Cloud", "https://api.groq.com/openai/v1/chat/completions",
        "openai/gpt-oss-120b", True,
        "API distante : vérifiez quota, prix et conditions du modèle choisi ; des limites ou une facturation peuvent s'appliquer.",
    ),
    "openrouter": Provider(
        "openrouter", "OpenRouter — modèles gratuits", "https://openrouter.ai/api/v1/chat/completions",
        "openrouter/free", True,
        "API distante : choisissez uniquement des modèles gratuits (:free) ; disponibilité et confidentialité dépendent du fournisseur.",
    ),
}


class GenerationError(RuntimeError):
    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


def provider_choices() -> list[Provider]:
    return list(PROVIDERS.values())


def _content_from_response(payload: dict) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "\n".join(
                str(item.get("text", "")) for item in content if isinstance(item, dict)
            ).strip()
    except (KeyError, IndexError, TypeError):
        pass
    # Ollama standard response
    message = payload.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"].strip()
    raise GenerationError("Réponse API vide ou de format inattendu.")


def _post_json(url: str, headers: dict[str, str], body: dict,
               timeout: float = 90.0) -> tuple[dict, dict[str, str]]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise GenerationError("Réponse trop volumineuse (> 8 Mo), ignorée.")
            return json.loads(raw.decode("utf-8")), {k.lower(): v for k, v in response.headers.items()}
    except urllib.error.HTTPError as exc:
        # Ne jamais recopier l'URL/headers, qui pourraient révéler la clé dans un journal.
        retryable = exc.code == 429 or 500 <= exc.code < 600
        detail = ""
        try:
            body_text = exc.read(1200).decode("utf-8", "replace")
            parsed = json.loads(body_text)
            detail = str(parsed.get("error", {}).get("message", ""))[:500]
        except Exception:
            pass
        if exc.code == 401 or exc.code == 403:
            raise GenerationError(f"Clé ou accès refusé (HTTP {exc.code}). Vérifiez la clé et les droits.") from None
        if exc.code == 429:
            raise GenerationError("Quota / limite de débit atteint (HTTP 429). Réessayez plus tard.", True) from None
        raise GenerationError(f"Le fournisseur a répondu HTTP {exc.code}. {detail}", retryable) from None
    except urllib.error.URLError as exc:
        raise GenerationError(
            f"Connexion impossible : {getattr(exc, 'reason', 'erreur réseau')}. "
            "Vérifiez Internet ou, pour Ollama, qu'il est démarré."
        , True) from None
    except (TimeoutError, OSError) as exc:
        raise GenerationError(f"Délai réseau dépassé ou connexion interrompue : {exc}", True) from None
    except json.JSONDecodeError:
        raise GenerationError("Le fournisseur a renvoyé une réponse JSON invalide.") from None


def generate_text(provider_key: str, model: str, prompt: str, *, api_key: str = "",
                  max_tokens: int = 1024, temperature: float = 0.8,
                  should_stop: Callable[[], bool] | None = None,
                  on_attempt: Callable[[], bool] | None = None,
                  retry_count: int = 2) -> str:
    """Génère un texte. Les retries 429/5xx sont bornés et ne masquent pas l'échec."""
    provider = PROVIDERS.get(provider_key)
    if provider is None:
        raise GenerationError("Fournisseur inconnu.")
    if provider.needs_key and not api_key.strip():
        raise GenerationError("Saisissez la clé API de ce fournisseur.")
    if not model.strip():
        raise GenerationError("Le nom du modèle est vide.")
    if provider_key == "openrouter" and not (
        model.strip() == "openrouter/free" or model.strip().endswith(":free")
    ):
        raise GenerationError(
            "Pour éviter tout coût imprévu, OpenRouter n'accepte ici que openrouter/free "
            "ou un identifiant de modèle se terminant par :free."
        )

    if provider_key == "ollama":
        url = provider.base_url
        headers = {"Content-Type": "application/json"}
        body = {
            "model": model.strip(),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"num_predict": max(64, min(4096, int(max_tokens))),
                         "temperature": max(0.0, min(1.5, float(temperature)))},
        }
    else:
        url = provider.base_url
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key.strip(),
        }
        if provider_key == "openrouter":
            headers["HTTP-Referer"] = "https://arena.ai/"
            headers["X-Title"] = "Genesis Studio"
        body = {
            "model": model.strip(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max(64, min(4096, int(max_tokens))),
            "temperature": max(0.0, min(1.5, float(temperature))),
        }

    attempts = max(0, min(3, int(retry_count)))
    for attempt in range(attempts + 1):
        if should_stop and should_stop():
            raise GenerationError("Génération annulée.")
        if on_attempt is not None and not on_attempt():
            raise GenerationError("Limite maximale de requêtes atteinte.")
        try:
            result, _headers = _post_json(url, headers, body)
            text = _content_from_response(result)
            if not text:
                raise GenerationError("Le modèle a renvoyé un texte vide.", True)
            return text
        except GenerationError as exc:
            if not exc.retryable or attempt >= attempts:
                raise
            # Courte attente exponentielle ; jamais de retry infini sur un quota épuisé.
            wait_s = min(8.0, 1.5 * (2 ** attempt))
            for _ in range(int(wait_s * 4)):
                if should_stop and should_stop():
                    raise GenerationError("Génération annulée pendant l'attente du quota.")
                time.sleep(0.25)
    raise GenerationError("Nombre de tentatives dépassé.")
