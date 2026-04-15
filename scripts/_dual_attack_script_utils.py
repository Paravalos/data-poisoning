"""Shared helpers for dual-attack command-line scripts."""

from __future__ import annotations

import os
import sys


def ensure_repo_root(script_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(script_path)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root
