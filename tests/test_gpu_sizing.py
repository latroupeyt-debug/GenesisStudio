"""Smoke tests CPU pour les chemins CUDA sûrs et le conseil de dimensionnement."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from genesis.core import gpu, sizing  # noqa: E402
from genesis.core.model import preset_model_config  # noqa: E402


def main() -> int:
    report = gpu.device_report()
    assert isinstance(report["torch"], str) and report["torch"]
    print("matériel:", report["label"], "· build CUDA:", report["cuda_build"])

    # Benchmark explicitement CPU : exécutable même sur une machine sans carte NVIDIA.
    result = gpu.run_benchmark(
        preset_model_config("nano", 512),
        device_pref="cpu",
        seq_len=32,
        max_batch=4,
        max_seconds=8,
    )
    assert not result.error, result.error
    assert result.device == "cpu"
    assert result.max_batch >= 1
    assert result.tokens_per_s > 0
    print("calibration:", result.summary())

    params = 1_000_000
    levels = [sizing.advise(params, int(params * ratio), epochs=2)
              for ratio in (0, 0.2, 5, 20, 100, 400)]
    assert [x.level for x in levels] == [
        "critique", "critique", "risque", "correct", "bon", "excellent"
    ]
    assert levels[3].ratio_corpus == 20
    assert levels[3].ratio_exposure == 40
    rows = sizing.preset_table(2048)
    assert [r["key"] for r in rows] == ["nano", "micro", "mini", "small"]
    assert all(r["tokens_comfort"] == 5 * r["tokens_min"] for r in rows)
    assert 18 < rows[0]["mb_min"] < 20  # Nano, ~4 caractères/token
    assert rows[0]["mb_comfort"] > rows[0]["mb_min"]
    print("dimensionnement:", ", ".join(f"{x.badge}:{x.ratio_corpus:g}" for x in levels))
    print("TOUS LES TESTS GPU / DIMENSIONNEMENT SONT PASSÉS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
