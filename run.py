#!/usr/bin/env python3
"""Lance Genesis Studio.

    python run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__" and "--smoke-test" in sys.argv:
    # Importe tout le graphe de démarrage sans ouvrir la fenêtre, pour vérifier
    # les bundles PyInstaller. Capture les erreurs pour retourner un code propre.
    try:
        from genesis.app import main as _main  # noqa: F401
    except Exception:
        import traceback

        traceback.print_exc()
        raise SystemExit(1)
    raise SystemExit(0)

from genesis.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
