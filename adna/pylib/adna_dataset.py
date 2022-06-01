import random
import sqlite3

import torch
from Bio.Seq import Seq
from sklearn.utils import class_weight
from torch.utils import data

from . import consts


def read_dataset(split="", *, limit=-1, db=None):
    """
    Args:
        split: "train", "val", "test". "" == all splits
        limit: Limit the dataset to this many records, -1 = all
        db:    Path to the database
    Returns:
        seqs, labels
        a list of sequences & a list of their labels
    """
    db = db if db else consts.SQL
    sql = "select seq, label, rev from seqs"
    args = []

    if split:
        sql += " where split = ? order by random()"
        args.append(split)

    if limit > 0:
        sql += " limit ?"
        args.append(limit)

    with sqlite3.connect(db) as cxn:
        seqs, labels = [], []
        for rec in cxn.execute(sql, args):
            seqs.append(rec[0])
            label = rec[1] + rec[2]
            labels.append(label)

    return seqs, labels


class ADnaDataset(data.Dataset):
    def __init__(
        self, split="", tokenizer=None, *, rev_comp_rate=0.0, to_n_rate=0.0, limit=-1
    ):
        """
        split         = "train", "val", "test" split from the database
                        "" = all splits
        tokenizer     = a previously trained Hugging Face BPE tokenizer
                        None = return raw sequence and label
        rev_comp_rate = convert sequences to its reverse complement at this random rate
        to_n_rate     = convert bases in a sequence to N at this random rate
        """
        self.split = split
        self.check_split()

        self.rev_comp_rate = rev_comp_rate
        self.to_n_rate = to_n_rate
        self.limit = limit

        self.tokenizer = tokenizer

        self.seqs, self.labels = read_dataset(self.split, limit=self.limit)

        self.weights = self.get_weights(self.labels)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        label = self.labels[idx]

        seq = self.rev_comp(seq)
        seq = self.to_n(seq)

        if not self.tokenizer:
            return seq

        encoded = self.tokenizer.encode_plus(
            seq, padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(label)
        return encoded

    def rev_comp(self, seq):
        if random.random() < self.rev_comp_rate:
            seq = str(Seq(seq).reverse_complement())
        return seq

    def to_n(self, seq):
        """Randomly convert bases to N."""
        if self.to_n_rate:
            bases = []
            for base in seq:
                if random.random() < self.to_n_rate:
                    bases.append("N")
                else:
                    bases.append(base)
            seq = "".join(bases)
        return seq

    @staticmethod
    def get_weights(labels):
        """Calculate the weight for each label."""
        classes = sorted(set(labels))
        wt = class_weight.compute_class_weight("balanced", classes=classes, y=labels)
        return wt

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
