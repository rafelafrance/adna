import sqlite3
from collections import namedtuple

import torch
from torch.utils import data

from . import consts


Record = namedtuple("Record", "seq label rev")


class ADnaDataset(data.Dataset):
    def __init__(self, split, tokenizer):
        """
        split = "train", "val", "test" from the database
        tokenizer = a previously trained Hugging Face BPE tokenizer
        """
        self.split = split
        self.check_split()

        self.tokenizer = tokenizer
        self.data = self.read_data()

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
            dataset = [Record(*r) for r in cxn.execute(sql, (self.split,))]
        return dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            record.seq, padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(record.label)
        return encoded

    def pos_weight(self):
        """Calculate the weight for the positive cases of the trait."""
        pos = sum(s.label for s in self.data)
        pos_wt = (len(self) - pos) / pos if pos > 0.0 else 1.0
        return [pos_wt]
