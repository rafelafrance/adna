import sqlite3

import torch
from torch.utils import data

from . import consts

# select *, row_number() over win as idx
# from seqs
# window win as (order by random())
# limit ?


class ADnaDataset(data.Dataset):
    def __init__(self, split, tokenizer):
        """
        split = "train", "val", "test" from the database
        tokenizer = a previously trained Hugging Face BPE tokenizer
        """
        self.split = split
        self.check_split()

        self.tokenizer = tokenizer

        self.data = []

    def check_split(self):
        with sqlite3.connect(consts.SQL) as cxn:
            rows = cxn.execute("select distinct split from seqs")
        splits = [r[0] for r in rows]
        if self.split not in splits:
            raise ValueError(
                f"Split {self.split} is not in the database.\nOptions are: {splits}"
            )

    def read_data(self):
        sql = "select seq, label, rev from seqs where split = ? order by random()"
        with sqlite3.connect(consts.SQL) as cxn:
            self.data = [r for r in cxn.execute(sql, (self.split,))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.data[idx][0], padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(self.data[idx][1])
        return encoded
