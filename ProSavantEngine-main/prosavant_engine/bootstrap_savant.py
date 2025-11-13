# bootstrap_savant.py
"""
Bootstrap module to ensure 'prosavant_engine' can be imported reliably
when running scripts directly (e.g. `python AGI_RRF_Phi9_Delta.py`).

Usage (at the very top of any entry script):

    import bootstrap_savant  # noqa: F401
    from prosavant_engine import AGIRRFCore

This works both in Colab and locally, even if you haven't done
`pip install -e .`, because it adds the repo root to sys.path.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _detect_repo_root() -> Path:
    """
    Walk upwards from this file until we find a directory that contains
    'prosavant_engine/'. That directory is treated as the repo root.
    """
    here = Path(__file__).resolve()

    for candidate in (here.parent, *here.parents):
        if (candidate / "prosavant_engine").is_dir():
            return candidate

    # Fallback: just use the directory that contains this file
    return here.parent


def add_repo_root_to_sys_path() -> str:
    """
    Ensure the detected repo root is on sys.path and return it.
    """
    repo_root = _detect_repo_root()
    root_str = str(repo_root)

    if root_str not in sys.path:
        # Put it at the front so it wins over site-packages
        sys.path.insert(0, root_str)

    # Optional: expose for debugging / downstream use
    os.environ.setdefault("PROSAVANT_ENGINE_ROOT", root_str)
    return root_str


# Run automatically on import
REPO_ROOT = add_repo_root_to_sys_path()
