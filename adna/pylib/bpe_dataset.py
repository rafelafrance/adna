import torch
from sklearn.utils import class_weight
from torch.utils import data

from . import consts
from . import dataset_utils as du


class BPEDataset(data.Dataset):
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
        limit         = limit the dataset to this many records, -1 = all records
        """
        du.check_split(split)
        self.tokenizer = tokenizer
        self.rev_comp_rate = rev_comp_rate
        self.to_n_rate = to_n_rate
        self.records = du.read_records(split, limit=limit)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        seq = record.seq
        seq = du.rev_comp(seq, self.rev_comp_rate)
        seq = du.to_n(seq, self.to_n_rate) if self.to_n_rate else seq

        encoded = self.tokenizer.encode_plus(
            seq, padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(record.label)
        return encoded

    @property
    def weights(self):
        """Calculate the weight for each label."""
        labels = [r.label for r in self.records]
        classes = sorted(set(labels))
        wt = class_weight.compute_class_weight("balanced", classes=classes, y=labels)
        return wt
