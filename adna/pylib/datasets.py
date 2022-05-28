import random
import sqlite3
from collections import namedtuple

import torch
from Bio.Seq import Seq
from sklearn.utils import class_weight
from torch.utils import data

from . import consts

REV_LABEL = 2  # If the seq in the DB is marked as rev give it this label
REV_COMP_LABELS = [0, 2, 1]  # How to convert labels when doing a reverse complement

Record = namedtuple("Record", "seq label rev")


class ADnaDataset(data.Dataset):
    def __init__(self, split="", tokenizer=None, rev_comp_rate=0.0, to_n_rate=0.0):
        """
        split         = "train", "val", "test" split from the database
                        "" = all splits
        tokenizer     = a previously trained Hugging Face BPE tokenizer
                        None = return raw sequence and label
        rev_comp_rate = convert sequences to its reverse complement at this random rate
        to_n_rate     = convert bases in a sequence to N at this random rate
        raw_seq       = If true, don't tokenize the sequence return the seq with label
        """
        self.split = split
        self.check_split()

        self.rev_comp_rate = rev_comp_rate
        self.to_n_rate = to_n_rate

        self.tokenizer = tokenizer
        self.records = self.read_data()

    def check_split(self):
        if not self.split:
            return
        with sqlite3.connect(consts.SQL) as cxn:
            rows = cxn.execute("select distinct split from seqs")
        splits = [r[0] for r in rows]
        if self.split not in splits:
            raise ValueError(
                f"Split {self.split} is not in the database.\nOptions are: {splits}"
            )

    def read_data(self):
        sql = "select seq, label, rev from seqs"
        args = []

        if self.split:
            sql += " where split = ? order by random()"
            args = (self.split,)

        with sqlite3.connect(consts.SQL) as cxn:
            dataset = [Record(*r) for r in cxn.execute(sql, args)]

        return dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq, label = self.get_fields(idx)
        seq, label = self.rev_comp(seq, label)
        seq = self.to_n(seq)

        if not self.tokenizer:
            return seq

        encoded = self.tokenizer.encode_plus(
            seq, padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(label)
        return encoded

    def get_fields(self, idx):
        record = self.records[idx]
        seq, label = record.seq, record.label
        label = REV_LABEL if record.rev else label
        return seq, label

    def rev_comp(self, seq, label):
        if random.random() < self.rev_comp_rate:
            seq = str(Seq(seq).reverse_complement())
            label = REV_COMP_LABELS[label]
        return seq, label

    def to_n(self, seq):
        if self.to_n_rate:
            bases = []
            for base in seq:
                if random.random() < self.to_n_rate:
                    bases.append("N")
                else:
                    bases.append(base)
            seq = "".join(bases)
        return seq

    def weights(self):
        """Calculate the weight for each label."""
        y = [r.label for r in self.records]
        classes = sorted(set(y))
        wt = class_weight.compute_class_weight("balanced", classes=classes, y=y)
        return wt
