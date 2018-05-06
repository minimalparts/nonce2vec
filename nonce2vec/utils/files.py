"""Files utils."""

import os
import smart_open

__all__ = ('Sentences', 'get_zipped_sentences', 'get_sentences',
           'get_model_path')


def get_model_path(datadir, outputdir):
    """Return absolute path to w2v model file.

    Model absolute path is computer from the outputdir and the
    datadir name.
    """
    os.makedirs(outputdir, exist_ok=True)
    return os.path.join(outputdir, '{}.w2v.model'.format(
        os.path.basename(datadir)))


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
    """An iterable class (with generators) for gensim and n2v."""

    def __init__(self, input_data, target):
        if target != 'gensim' and target != 'n2v':
            raise Exception('Invalid target parameter \'{}\''.format(target))
        self._target = target
        if target == 'gensim':
            self._datadir = input_data
        if target == 'n2v':
            self._datafile = input_data

    def _iterate_for_gensim(self):
        for filename in os.listdir(self._datadir):
            if filename.startswith('.'):
                continue
            with open(os.path.join(self._datadir, filename), 'rt') \
             as input_stream:
                for line in input_stream:
                    yield line.strip().split()

    def _iterate_for_n2v(self):
        with open(self._datafile, 'rt') as input_stream:
            for line in input_stream:
                yield line.rstrip('\n').split('\t')

    def __iter__(self):
        if self._target == 'gensim':
            return self._iterate_for_gensim()
        if self._target == 'n2v':
            return self._iterate_for_n2v()
        raise Exception('Invalid target parameter \'{}\''.format(self._target))
