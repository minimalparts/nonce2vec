"""Files utils."""

import os
import smart_open

__all__ = ('Sentences', 'get_zipped_sentences', 'get_sentences')


def get_zipped_sentences(datazip):
    """Return generator over sentence split of wikidump for gensim.word2vec.

    datazip should be the absolute path to the wikidump.gzip file.
    """
    for filename in os.listdir(smart_open.smart_open(datazip)):
        with open(filename, 'r') as input_stream:
            for line in input_stream:
                yield line.strip().split()


def get_sentences(data):
    """Return generator over sentence split of wikidata for gensim.word2vec."""
    for filename in os.listdir(data):
        if filename.startswith('.'):
            continue
        with open(os.path.join(data, filename), 'r') as input_stream:
            for line in input_stream:
                yield line.strip().split()


class Sentences(object):
    """An iterable class (with generators) for gensim."""

    def __init__(self, datadir):
        self.datadir = datadir

    def __iter__(self):
        for filename in os.listdir(self.datadir):
            if filename.startswith('.'):
                continue
            with open(os.path.join(self.datadir, filename), 'r') \
             as input_stream:
                for line in input_stream:
                    yield line.strip().split()
