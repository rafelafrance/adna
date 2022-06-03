from pathlib import Path

# #########################################################################
IS_SUBDIR = Path.cwd().name in ("notebooks", "experiments")
ROOT_DIR = Path(".." if IS_SUBDIR else ".")

DATA_DIR = ROOT_DIR / "data"

# ######### Model Params #################################################
PREFIX = "UF46992"

VOCAB_SIZE = 4096
MAX_LENGTH = 100  # Pad/truncate all sequences to this tokenized length

# #########################################################################
# Special tokens

BOS = "<s>"  # Beginning of sequence -- Also holds inferred class label
EOS = "</s>"  # End of sequence
PAD = "<pad>"  # Sequence padding to get to a uniform SEQ_LENGTH
UNK = "<unk>"  # Unknown token -- Currently unused
MASK = "<mask>"  # Mask a token -- Currently unused

SPECIAL_TOKENS = [BOS, PAD, EOS, UNK, MASK]

# #########################################################################
MT_DIR = DATA_DIR / PREFIX
SQL = MT_DIR / f"{PREFIX}.sqlite"
