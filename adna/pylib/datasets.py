import sqlite3
from collections import namedtuple

import torch
from sklearn.utils import class_weight
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
        self.records = self.read_data()

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
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        encoded = self.tokenizer.encode_plus(
            record.seq, padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(record.label)
        return encoded

    def weights(self):
        """Calculate the weight for each label."""
        y = [r.label for r in self.records]
        classes = sorted(set(y))
        wt = class_weight.compute_class_weight('balanced', classes=classes, y=y)
        return wt
