"""Régressions de configuration PyInstaller détectables sans build Windows."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "genesis.spec"


def main() -> int:
    tree = ast.parse(SPEC.read_text(encoding="utf-8"), filename=str(SPEC))
    analysis = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "Analysis"
    )
    keywords = {keyword.arg: keyword.value for keyword in analysis.keywords}
    excludes_node = keywords["excludes"]
    excludes = {item.value for item in excludes_node.elts if isinstance(item, ast.Constant)}
    assert "torch.distributed" not in excludes, "torch.distributed ne doit pas être exclu."
    assert "torch.testing" not in excludes, "torch.testing est importé pendant l'initialisation de PyTorch."

    hiddenimports = ast.unparse(keywords["hiddenimports"])
    assert "collect_submodules" in hiddenimports and "torch.distributed" in hiddenimports, (
        "PyInstaller doit collecter explicitement tous les sous-modules torch.distributed."
    )
    assert "torch.testing" in hiddenimports, "PyInstaller doit inclure torch.testing."
    print("Configuration PyInstaller : torch.distributed et ses sous-modules sont collectés — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
