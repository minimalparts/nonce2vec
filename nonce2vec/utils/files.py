"""Files utils."""

import os
import smart_open

__all__ = ('Samples', 'get_zipped_sentences', 'get_sentences',
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


class Samples(object):
    """An iterable class (with generators) for gensim and n2v."""

    def __init__(self, input_data, source):
        if source != 'wiki' and source != 'nonces' and source != 'chimeras':
            raise Exception('Invalid source parameter \'{}\''.format(source))
        self._source = source
        if source == 'wiki':
            self._datadir = input_data
        if source == 'nonces' or source == 'chimeras':
            self._datafile = input_data

    def _iterate_over_wiki(self):
        for filename in os.listdir(self._datadir):
            if filename.startswith('.'):
                continue
            with open(os.path.join(self._datadir, filename), 'rt') \
             as input_stream:
                for line in input_stream:
                    yield line.strip().split()

    def _iterate_over_nonces(self):
        with open(self._datafile, 'rt') as input_stream:
            for line in input_stream:
                fields = line.rstrip('\n').split('\t')
                nonce = fields[0]
                sentence = fields[1].replace('___', nonce).split()
                probe = '{}_true'.format(nonce)
                yield [sentence], nonce, probe

    def _iterate_over_chimeras(self):
        with open(self._datafile, 'rt') as input_stream:
            for line in input_stream:
                fields = line.rstrip('\n').split('\t')
                sentences = []
                for sent in fields[1].split('@@'):
                    tokens = sent.strip().split(' ')
                    if '___' in tokens:
                        sentences.append(tokens)
                probes = fields[2].split(',')
                responses = fields[3].split(',')
                yield sentences, probes, responses

    def __iter__(self):
        if self._source == 'wiki':
            return self._iterate_over_wiki()
        if self._source == 'nonces':
            return self._iterate_over_nonces()
        if self._source == 'chimeras':
            return self._iterate_over_chimeras()
        raise Exception('Invalid source parameter \'{}\''.format(self._source))
