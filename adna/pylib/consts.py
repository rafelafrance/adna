"""Literals used in the system."""
from pathlib import Path

# #########################################################################
IS_SUBDIR = Path.cwd().name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"
