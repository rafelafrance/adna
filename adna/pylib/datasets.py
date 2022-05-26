import sqlite3

import torch
from torch.utils import data

from . import consts

# select *, row_number() over win as idx
# from seqs
# window win as (order by random())
# limit ?


class ADnaDataset(data.Dataset):
    def __init__(self, split, tokenizer, limit=-1):
        """
        split = "train", "val", "test" from the database
        tokenizer = a previously trained Hugging Face BPE tokenizer
        limit = limit the records in the dataset. -1 == no limit
        """
        self.split = split
        self.check_split()

        self.tokenizer = tokenizer
        self.limit = limit

        self.length = 0
        self.cxn = sqlite3.connect(":memory:")
        self.sample_data()

    def check_split(self):
        with sqlite3.connect(consts.SQL) as cxn:
            rows = cxn.execute("select distinct split from seqs")
        splits = [r[0] for r in rows]
        if self.split not in splits:
            raise ValueError(
                f"Split {self.split} is not in the database.\nOptions are: {splits}"
            )

    def sample_data(self):
        create = """
            create table seqs as
            select seq, label, rev
            from aux.seqs where split = ?
            order by random()
            limit ?;
            """
        self.cxn.execute("drop table if exists seqs;")
        self.cxn.execute(f"attach database '{consts.SQL}' as aux")
        self.cxn.execute(create, (self.split, self.limit))
        self.cxn.execute("detach database aux")
        count = self.cxn.execute("select count(*) from seqs")
        self.length = count.fetchone()[0]

    def __len__(self):
        return self.length

    def __getitem__(self, row_id):
        row_id += 1
        sql = "select * from seqs where rowid = ?"
        row = self.cxn.execute(sql, (row_id,)).fetchone()
        encoded = self.tokenizer.encode_plus(
            row[0], padding="max_length", max_length=consts.MAX_LENGTH
        )
        encoded["label"] = torch.tensor(row[1])
        return encoded
