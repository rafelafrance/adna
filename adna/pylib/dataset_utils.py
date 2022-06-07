import random
import sqlite3
from collections import namedtuple

from Bio.Seq import Seq

from . import consts

SeqRecord = namedtuple("SeqRecord", "seq label")


def read_records(split="", *, limit=-1, db=None):
    """
    Args:
        split: "train", "val", "test". "" == all splits
        limit: Limit the dataset to this many records, -1 = all
        db:    Path to the database
    Returns:
        a list of sequence records [SeqRecord]
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
        records = []
        for rec in cxn.execute(sql, args):
            label = rec[1] + rec[2]
            records.append(SeqRecord(rec[0], label))

    return records


def read_seqs_labels(split="", *, limit=-1, db=None):
    records = read_records(split, limit=limit, db=db)
    seqs, labels = [], []
    for rec in records:
        seqs.append(rec.seq)
        labels.append(rec.label)
    return seqs, labels


def rev_comp(seq, rate=1.0):
    """Randomly convert a sequence to its reverse complement."""
    if random.random() < rate:
        seq = str(Seq(seq).reverse_complement())
    return seq


def to_n(seq, rate=0.0):
    """Randomly convert bases to N."""
    bases = []
    for base in seq:
        if random.random() < rate:
            bases.append("N")
        else:
            bases.append(base)
    seq = "".join(bases)
    return seq


def check_split(split):
    if not split:
        return
    with sqlite3.connect(consts.SQL) as cxn:
        rows = cxn.execute("select distinct split from seqs")
    splits = [r[0] for r in rows]
    if split not in splits:
        raise ValueError(
            f"Split {split} is not in the database.\nOptions are: {splits}"
        )
