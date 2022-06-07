import random

import torch
from torch.utils import data

from . import consts
from . import dataset_utils as du


class BalancedDataset(data.Dataset):
    def __init__(self, split="", tokenizer=None, *, to_n_rate=0.0, limit=-1):
        """
        Under-sample the negative samples to create a balanced dataset.

        split         = "train", "val", "test" split from the database
                        "" = all splits
        tokenizer     = a previously trained Hugging Face BPE tokenizer
                        None = return raw sequence and label
        to_n_rate     = convert bases in a sequence to N at this random rate
        limit         = limit the dataset to this many records, -1 = all records
        """
        du.check_split(split)
        self.to_n_rate = to_n_rate
        self.tokenizer = tokenizer
        self.weights = [1.0, 1.0]

        records = du.read_records(split, limit=limit)
        records = self.rev_comp_all(records)
        pos_records, neg_records = self.pos_neg_records(records)
        records = pos_records + random.sample(neg_records, k=len(pos_records))
        self.records = random.sample(records, k=len(records))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        seq = record.seq
        seq = du.to_n(seq, self.to_n_rate) if self.to_n_rate else seq

        encoded = self.tokenizer.encode_plus(
            seq, padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(record.label)
        return encoded

    @staticmethod
    def rev_comp_all(records):
        new_recs = []
        for rec in records:
            seq = du.rev_comp(rec.seq)
            new_recs.append(du.SeqRecord(seq, rec.label))
        records += new_recs
        return records

    @staticmethod
    def pos_neg_records(records):
        """Split the records into positive and negative records."""
        pos_recs, neg_recs = [], []
        for rec in records:
            if rec.label == 1:
                pos_recs.append(rec)
            else:
                neg_recs.append(rec)
        return pos_recs, neg_recs
