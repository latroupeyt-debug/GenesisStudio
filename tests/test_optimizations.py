"""Tests d'attention SDPA et repli torch.compile sur CPU."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis.core import gpu  # noqa: E402
from genesis.core.model import CausalSelfAttention, GPT, ModelConfig  # noqa: E402
from genesis.core.trainer import TrainConfig, TrainHooks, Trainer  # noqa: E402


def test_sdpa_matches_manual_and_is_causal() -> None:
    torch.manual_seed(7)
    cfg = ModelConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2,
                      n_embd=8, dropout=0.0)
    attention = CausalSelfAttention(cfg).eval()
    x = torch.randn(2, 8, 8)
    attention.use_sdpa = True
    fast = attention(x)
    attention.use_sdpa = False
    manual = attention(x)
    torch.testing.assert_close(fast, manual, rtol=1e-5, atol=1e-6)

    # Modifier les tokens futurs ne doit pas modifier les sorties antérieures.
    changed = x.clone()
    changed[:, 5:, :] = torch.randn_like(changed[:, 5:, :]) * 100
    torch.testing.assert_close(attention(x)[:, :5], attention(changed)[:, :5],
                               rtol=1e-5, atol=1e-6)


class FailsOnce(nn.Module):
    """Simule un backend Triton/Inductor absent sur le premier appel compilé."""

    def __init__(self, original: nn.Module):
        super().__init__()
        self._orig_mod = original

    def forward(self, *args, **kwargs):
        raise RuntimeError("simulated torch.compile backend unavailable")


def test_compile_falls_back_without_checkpoint_prefix() -> None:
    with tempfile.TemporaryDirectory(prefix="genesis-compile-") as temp:
        root = Path(temp)
        token_path = root / "tokens.bin"
        np.arange(1024, dtype=np.uint16).__mod__(16).tofile(token_path)
        cfg = ModelConfig(vocab_size=16, block_size=16, n_layer=1,
                          n_head=2, n_embd=8, dropout=0.0)
        logs: list[str] = []
        hooks = TrainHooks(on_log=logs.append)
        train_cfg = TrainConfig(
            epochs=1, batch_size=1, seq_len=8, lr=1e-3, warmup=0,
            eval_every=0, ckpt_every=0, max_steps=1, device="cpu",
            compile_model=True, threads=1,
        )
        trainer = Trainer(
            token_path=token_path, dtype="uint16", n_tokens=1024,
            model=GPT(cfg), train_cfg=train_cfg,
            ckpt_last=root / "last.pt", ckpt_best=root / "best.pt", hooks=hooks,
        )
        with patch("torch.compile", side_effect=lambda model, **kwargs: FailsOnce(model)):
            result = trainer.run()
        assert result["steps"] == 1
        assert any("repli automatique" in line or "Compilation échouée" in line for line in logs), logs
        payload = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        assert not any("_orig_mod" in key for key in payload["model_state"])
        assert payload["model_config"]["n_embd"] == 8


def main() -> int:
    # Compatibilité avec les anciennes versions torch.amp (>= 2.1).
    with patch("torch.amp.GradScaler", side_effect=TypeError("ancienne API")):
        scaler = gpu.make_grad_scaler(enabled=False)
        assert not scaler.is_enabled()
    test_sdpa_matches_manual_and_is_causal()
    print("SDPA : équivalence au chemin manuel + masque causal OK")
    test_compile_falls_back_without_checkpoint_prefix()
    print("torch.compile : repli eager + checkpoint portable OK")
    print("TOUS LES TESTS D'OPTIMISATION SONT PASSÉS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
