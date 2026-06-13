"""Sanity check that the dev environment is wired up correctly.

Feel free to extend this file with real tests for your solution, or delete it
and start from scratch — it exists so that `make check-all` passes on a fresh
clone before any candidate code is written.
"""

from __future__ import annotations

import sys


def test_python_version() -> None:
    assert sys.version_info >= (3, 13), "This repo requires Python 3.13+"


def test_pandas_importable() -> None:
    import pandas as pd

    assert pd.__version__
