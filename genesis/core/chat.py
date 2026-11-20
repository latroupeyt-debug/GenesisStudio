"""Discussion avec un modèle maison : gabarits de prompt + génération en flux."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

import torch

from .model import GPT
from .tokenizers import SPECIAL, Tokenizer

INSTRUCTION_TEMPLATE = "### Instruction :\n{user}\n\n### Réponse :\n"
DIALOGUE_TEMPLATE = "Utilisateur : {user}\nAssistant :"

PROMPT_TEMPLATES: dict[str, dict] = {
    "continuation": {
        "label": "Continuation libre",
        "description": "Le modèle prolonge simplement votre texte. C'est le mode le plus "
                       "naturel pour un modèle entraîné sur un corpus brut.",
        "template": "{user}",
        "stops": [SPECIAL],
    },
    "instruction": {
        "label": "Instruction / Réponse",
        "description": "Ajoute des balises « ### Instruction / ### Réponse ». Utile si votre "
                       "corpus contient des paires question-réponse.",
        "template": INSTRUCTION_TEMPLATE,
        "stops": ["### Instruction", SPECIAL, "\n\n### "],
    },
    "dialogue": {
        "label": "Dialogue (Utilisateur / Assistant)",
        "description": "Format conversation simple. À privilégier si votre corpus est fait de "
                       "dialogues ou de transcriptions.",
        "template": DIALOGUE_TEMPLATE,
        "stops": ["Utilisateur :", "Utilisateur:", SPECIAL],
    },
}


@dataclass
class SamplingParams:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.95
    max_new_tokens: int = 160
    repetition_penalty: float = 1.1
    seed: int = -1

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature, "top_k": self.top_k, "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "repetition_penalty": self.repetition_penalty,
        }


def build_prompt(template_key: str, text: str, custom_template: str | None = None) -> str:
    spec = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["continuation"])
    template = custom_template or spec["template"]
    try:
        return template.format(user=text)
    except (KeyError, IndexError):
        return text


def stop_strings(template_key: str, custom: list[str] | None = None) -> list[str]:
    if custom is not None:
        return custom
    spec = PROMPT_TEMPLATES.get(template_key, PROMPT_TEMPLATES["continuation"])
    return list(spec["stops"])


def stream_generate(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    params: SamplingParams,
    on_text: Callable[[str], None],
    should_stop: Callable[[], bool] | None = None,
    stops: list[str] | None = None,
    device: torch.device | None = None,
) -> str:
    """Génère la réponse token par token et appelle on_text à chaque nouveau morceau."""
    device = device or next(model.parameters()).device
    if params.seed >= 0:
        torch.manual_seed(params.seed)

    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [tokenizer.encode(" ")[0] if tokenizer.encode(" ") else 0]
    ids = ids[-model.cfg.block_size :]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    stops = stops if stops is not None else []
    emitted = ""
    generated_ids: list[int] = []
    special_id = 0

    model.eval()
    with torch.no_grad():
        for _ in range(max(1, params.max_new_tokens)):
            if should_stop and should_stop():
                break
            next_id = model.sample_next(
                idx,
                temperature=params.temperature,
                top_k=params.top_k or None,
                top_p=params.top_p or None,
                repetition_penalty=params.repetition_penalty,
            )
            if next_id == special_id:
                break
            generated_ids.append(next_id)
            idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if idx.size(1) > model.cfg.block_size:
                idx = idx[:, -model.cfg.block_size :]

            full = tokenizer.decode(generated_ids)
            # un caractère multi-octets peut être incomplet : on attend l'octet suivant
            safe = full.rstrip("\ufffd") if full.endswith("\ufffd") else full
            if len(safe) > len(emitted):
                piece = safe[len(emitted):]
                if any(stop and stop in safe for stop in stops):
                    for stop in stops:
                        cut = safe.find(stop) if stop else -1
                        if cut >= 0:
                            piece = safe[len(emitted):cut]
                            if piece:
                                on_text(piece)
                            return safe[:cut]
                emitted = safe
                on_text(piece)
    final = tokenizer.decode(generated_ids)
    if len(final) > len(emitted):
        on_text(final[len(emitted):])
        emitted = final
    return emitted


class StopFlag:
    """Petit drapeau thread-safe pour interrompre une génération."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        self._event.set()

    def clear(self) -> None:
        self._event.clear()

    def __call__(self) -> bool:
        return self._event.is_set()
