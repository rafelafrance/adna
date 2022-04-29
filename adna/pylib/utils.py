"""Misc. utilities."""
import bz2
import gzip
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def open_file(file_name):
    path = Path(file_name)
    if path.suffix in ('.gz', ".gzip"):
        stream = gzip.open(file_name, 'rt')
    elif path.suffix in (".bzip", ".bz2"):
        stream = bz2.open(file_name, 'rt')
    else:
        stream = open(file_name)

    try:
        yield stream
    finally:
        stream.close()

