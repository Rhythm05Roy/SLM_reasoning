#!/usr/bin/env python3
"""
curricsym/train.py — Shim entry-point
======================================
Allows running the pipeline from inside the package directory:

    cd /path/to/SLM_reasoning
    python curricsym/train.py [OPTIONS]

All logic lives in the root-level train.py; this file simply re-executes
it so both invocation styles work without duplicating code.

Thesis note (Code Organization §3):
    The canonical entry point is the project-root train.py.
    This shim satisfies the convention of running scripts from inside
    the package directory without any import-path juggling.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Re-execute the canonical train.py as __main__
runpy.run_path(str(_ROOT / "train.py"), run_name="__main__")
