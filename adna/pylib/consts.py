"""Literals used in the system."""
from pathlib import Path

# #########################################################################
IS_SUBDIR = Path.cwd().name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"

# ######### Model Params #################################################
PREFIX = 'UF46992'

VOCAB_SIZE = 4096
MIN_FREQ = 2
MAX_LENGTH = 80  # Pad all sequences to this length
VECTOR_SIZE = 128  # Keeping things small for now

# #########################################################################
# Special tokens

BOS = "<s>"  # Beginning of sequence -- Also holds inferred class label
EOS = "</s>"  # End of sequence
PAD = "<pad>"  # Sequence padding to get to a uniform SEQ_LENGTH
UNK = "<unk>"  # Unknown token -- Currently unused
MASK = "<mask>"  # Mask a token -- Currently unused

SPECIAL_TOKENS = [BOS, PAD, EOS, UNK, MASK]

# #########################################################################
SUB_DIR = DATA_DIR / PREFIX
SQL = SUB_DIR / f'{PREFIX}.sqlite'

# select *, row_number() over win as idx
# from seqs window win as (order by random()) limit 1000000
