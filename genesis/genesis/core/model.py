"""Un petit GPT écrit de zéro (PyTorch), pensé pour tourner sur un PC normal.

Architecture volontairement classique et lisible :
    tokens -> embeddings (mots + positions) -> N blocs Transformer
           -> LayerNorm -> projection sur le vocabulaire -> softmax

Aucun code copié d'une bibliothèque de haut niveau : tout est ici, ce qui rend
les réglages (nombre de couches, largeur, tête d'attention…) transparents.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration & gabarits proposés à l'utilisateur
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    vocab_size: int = 64
    block_size: int = 128      # longueur de contexte (tokens)
    n_layer: int = 2           # nombre de blocs Transformer
    n_head: int = 2            # têtes d'attention par bloc
    n_embd: int = 64           # largeur du modèle
    dropout: float = 0.0
    bias: bool = False
    tie_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def validate(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError("La largeur du modèle doit être divisible par le nombre de têtes d'attention.")


PRESETS: list[dict] = [
    {
        "key": "nano",
        "label": "Nano",
        "tagline": "Pour découvrir, en quelques minutes",
        "n_layer": 2, "n_head": 2, "n_embd": 64, "block_size": 128, "dropout": 0.0,
        "description": (
            "Le plus petit gabarit. Il apprend la structure d'une langue, la ponctuation et "
            "des enchaînements courts. Idéal pour un premier essai rapide sur un petit corpus "
            "(quelques centaines de Ko). Entraînement en quelques minutes sur un CPU."
        ),
        "expect": "Phrases courtes, souvent grammaticales, sans véritable raisonnement.",
        "speed": "très rapide",
        "vram": "aucune (CPU suffit)",
    },
    {
        "key": "micro",
        "label": "Micro",
        "tagline": "Le bon compromis pour commencer",
        "n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 256, "dropout": 0.1,
        "description": (
            "Le gabarit recommandé pour un premier vrai modèle. Il commence à mémoriser des "
            "formules, des styles et des enchaînements de plusieurs phrases. Entraînement de "
            "quelques dizaines de minutes à quelques heures selon le corpus."
        ),
        "expect": "Texte cohérent sur 1 à 3 phrases, style reconnaissable.",
        "speed": "rapide",
        "vram": "~1 Go",
    },
    {
        "key": "mini",
        "label": "Mini",
        "tagline": "Pour un style vraiment reconnaissable",
        "n_layer": 6, "n_head": 6, "n_embd": 256, "block_size": 256, "dropout": 0.1,
        "description": (
            "Plus de profondeur : le modèle capte des dépendances plus longues et un style "
            "plus net. Il demande un corpus plus gros (plusieurs Mo) pour ne pas sur-apprendre. "
            "Un GPU accélère nettement l'entraînement."
        ),
        "expect": "Paragraphes cohérents, ton fidèle au corpus, encore des incohérences.",
        "speed": "moyen",
        "vram": "~2 Go",
    },
    {
        "key": "small",
        "label": "Small",
        "tagline": "Gros modèle maison (patience requise)",
        "n_layer": 6, "n_head": 8, "n_embd": 384, "block_size": 512, "dropout": 0.1,
        "description": (
            "Un vrai petit modèle de recherche. Excellent si vous avez plusieurs dizaines de "
            "Mo de texte et du temps (ou un GPU). Sur CPU, comptez plusieurs heures voire "
            "plusieurs jours : préférez reprendre l'entraînement par étapes."
        ),
        "expect": "Textes longs et structurés si le corpus est important.",
        "speed": "lent sur CPU",
        "vram": "~4 Go",
    },
    {
        "key": "custom",
        "label": "Personnalisé",
        "tagline": "Vous réglez chaque paramètre",
        "n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 256, "dropout": 0.1,
        "description": (
            "Choisissez vous-même la profondeur (couches), le nombre de têtes, la largeur "
            "(dimension des vecteurs) et la longueur de contexte. Le nombre de paramètres est "
            "recalculé en direct. Règle d'or : largeur divisible par le nombre de têtes."
        ),
        "expect": "Dépend entièrement de vos réglages.",
        "speed": "variable",
        "vram": "variable",
    },
]

PRESETS_BY_KEY = {p["key"]: p for p in PRESETS}


def preset_model_config(preset_key: str, vocab_size: int) -> dict:
    preset = PRESETS_BY_KEY.get(preset_key, PRESETS_BY_KEY["micro"])
    cfg = ModelConfig(
        vocab_size=vocab_size,
        block_size=int(preset["block_size"]),
        n_layer=int(preset["n_layer"]),
        n_head=int(preset["n_head"]),
        n_embd=int(preset["n_embd"]),
        dropout=float(preset["dropout"]),
    )
    data = cfg.to_dict()
    data["preset"] = preset_key
    data["preset_label"] = preset["label"]
    return data


def count_parameters(cfg: ModelConfig) -> int:
    """Nombre de paramètres, calculé à la main (sans instancier le modèle)."""
    e, L, V, T = cfg.n_embd, cfg.n_layer, cfg.vocab_size, cfg.block_size
    per_layer = 12 * e * e + (13 * e if cfg.bias else 0)
    total = L * per_layer + 2 * e  # + LayerNorm final
    total += V * e + T * e         # embeddings tokens + positions
    if not cfg.tie_weights:
        total += V * e
    return total


def human_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f} Md"
    if n >= 1_000_000:
        return f"{n / 1e6:.2f} M"
    if n >= 1_000:
        return f"{n / 1e3:.0f} k"
    return str(n)


def estimate_checkpoint_mb(n_params: int) -> float:
    # poids (float32) + état de l'optimiseur AdamW (2 moments)
    return n_params * 4 * 3 / (1024 ** 2)


# ---------------------------------------------------------------------------
# Briques du Transformer
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.attn_dropout = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)
        self._sdpa = hasattr(F, "scaled_dot_product_attention")
        self.use_sdpa = True  # le backend PyTorch choisit Flash/efficient/math selon le GPU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        if self._sdpa and self.use_sdpa:
            y = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=True,
            )
        else:  # repli manuel : masque causal explicite
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = F.dropout(att, p=self.attn_dropout, training=self.training)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))   # pré-normalisation + connexion résiduelle
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(cfg.vocab_size, cfg.n_embd),
            "wpe": nn.Embedding(cfg.block_size, cfg.n_embd),
            "drop": nn.Dropout(cfg.dropout),
            "h": nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            "ln_f": nn.LayerNorm(cfg.n_embd, bias=cfg.bias),
        })
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.transformer["wte"].weight
        self.apply(self._init_weights)
        # initialisation plus discrète des projections résiduelles
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer["wpe"].weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"Séquence trop longue ({T} > {self.cfg.block_size}).")
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer["drop"](self.transformer["wte"](idx) + self.transformer["wpe"](pos))
        for block in self.transformer["h"]:
            x = block(x)
        x = self.transformer["ln_f"](x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
            return logits, loss
        # en inférence, on ne calcule que le dernier pas de temps
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    def configure_optimizer(self, lr: float, weight_decay: float, betas=(0.9, 0.95)) -> torch.optim.Optimizer:
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8)

    # -- génération -----------------------------------------------------
    @torch.no_grad()
    def sample_next(self, idx: torch.Tensor, temperature: float = 0.8,
                    top_k: int | None = None, top_p: float | None = None,
                    repetition_penalty: float = 1.0) -> int:
        idx_cond = idx[:, -self.cfg.block_size :]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :].float()

        if repetition_penalty and repetition_penalty != 1.0:
            recent = idx_cond[0, -64:].tolist()
            for token in set(recent):
                if logits[0, token] > 0:
                    logits[0, token] /= repetition_penalty
                else:
                    logits[0, token] *= repetition_penalty

        if temperature <= 0:
            return int(torch.argmax(logits, dim=-1).item())
        logits = logits / max(temperature, 1e-5)

        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            kth = torch.topk(logits, k).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if top_p and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(probs, dim=-1)
            remove = cumulative - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        stop_ids: set[int] | None = None,
        on_token: Callable[[int], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> list[int]:
        out: list[int] = []
        for _ in range(max_new_tokens):
            if should_stop and should_stop():
                break
            next_id = self.sample_next(idx, temperature, top_k, top_p, repetition_penalty)
            idx = torch.cat([idx, torch.tensor([[next_id]], dtype=torch.long, device=idx.device)], dim=1)
            out.append(next_id)
            if on_token:
                on_token(next_id)
            if stop_ids and next_id in stop_ids:
                break
        return out


def pick_device(preference: str = "auto") -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device: torch.device) -> str:
    if device.type == "cuda":
        try:
            return f"GPU · {torch.cuda.get_device_name(0)}"
        except Exception:
            return "GPU"
    if device.type == "mps":
        return "GPU Apple (MPS)"
    return "Processeur (CPU)"


# ---------------------------------------------------------------------------
# Explications générées à partir de la configuration réelle du modèle
# ---------------------------------------------------------------------------
def architecture_text(cfg: ModelConfig) -> str:
    """Description lisible de l'architecture effective."""
    params = count_parameters(cfg)
    head_dim = cfg.n_embd // max(1, cfg.n_head)
    return (
        f"{cfg.n_layer} bloc(s) Transformer · {cfg.n_head} tête(s) d'attention de "
        f"{head_dim} dimensions · largeur {cfg.n_embd} · contexte {cfg.block_size} tokens · "
        f"vocabulaire {cfg.vocab_size} · {human_params(params)} paramètres "
        f"({params:,} au total), soit environ {estimate_checkpoint_mb(params):.1f} Mo sur disque."
    ).replace(",", " ")


def architecture_code(cfg: ModelConfig) -> str:
    """Aperçu du code réellement exécuté pour cette architecture."""
    head_dim = cfg.n_embd // max(1, cfg.n_head)
    return f'''# Extrait de genesis/core/model.py — instanciation réelle
cfg = ModelConfig(
    vocab_size={cfg.vocab_size},   # appris sur votre corpus par le tokenizer
    block_size={cfg.block_size},   # longueur de contexte (tokens)
    n_layer={cfg.n_layer},         # profondeur
    n_head={cfg.n_head},           # têtes d'attention (head_dim={head_dim})
    n_embd={cfg.n_embd},           # largeur des vecteurs
    dropout={cfg.dropout},
    bias={cfg.bias},
    tie_weights={cfg.tie_weights},  # poids partagés entrée/sortie
)
model = GPT(cfg)                   # -> {count_parameters(cfg):,} paramètres
optimizer = model.configure_optimizer(lr=..., weight_decay=...)

# Chaque bloc exécute :
#   x = x + Attention(LayerNorm(x))     # attention causale multi-têtes
#   x = x + MLP(LayerNorm(x))           # 2 couches denses + GELU (x4)
# Puis : LayerNorm finale + projection sur le vocabulaire
'''
