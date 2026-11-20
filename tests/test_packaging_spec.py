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
    excludes_node = next(keyword.value for keyword in analysis.keywords if keyword.arg == "excludes")
    excludes = {item.value for item in excludes_node.elts if isinstance(item, ast.Constant)}
    assert "torch.distributed" not in excludes, (
        "torch.distributed est importé pendant l'initialisation de torch.utils.data; "
        "l'exclure casse le démarrage de l'application empaquetée."
    )
    print("Configuration PyInstaller : torch.distributed n'est pas exclu — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
