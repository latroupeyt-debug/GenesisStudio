#!/usr/bin/env python3
"""Lance Genesis Studio.

    python run.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from genesis.app import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
