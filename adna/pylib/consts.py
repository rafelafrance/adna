"""Literals used in the system."""
from pathlib import Path

# #########################################################################
IS_SUBDIR = Path.cwd().name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"

# ######### Model Params #################################################
PREFIX = 'UF46992'

VOCAB_SIZE = 5000
MIN_FREQ = 2
SEQ_LENGTH = 80
VECTOR_SIZE = 128  # Keeping things small for now

# #########################################################################
SUB_DIR = DATA_DIR / PREFIX
SQL = SUB_DIR / f'{PREFIX}.sqlite'
