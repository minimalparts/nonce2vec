"""Files utils."""

import os
import random
import logging

__all__ = ('Samples', 'get_model_path')

logger = logging.getLogger(__name__)

DATASETS = {
    'men': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'resources', 'MEN_dataset_natural_form_full'),
    'def': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'resources', 'nonce.definitions.299.test'),
    'l2': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'resources', 'chimeras.dataset.l2.tokenised.test.txt'),
    'l4': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'resources', 'chimeras.dataset.l4.tokenised.test.txt'),
    'l6': os.path.join(os.path.dirname(os.path.dirname(__file__)),
                       'resources', 'chimeras.dataset.l6.tokenised.test.txt')
}


def get_model_path(datadir, outputdir, train_mode, alpha, neg, window_size,
                   sample, epochs, min_count, size):
    """Return absolute path to w2v model file.

    Model absolute path is computed from the outputdir and the
    datadir name.
    """
    os.makedirs(outputdir, exist_ok=True)
    return os.path.join(
        outputdir,
        '{}.{}.alpha{}.neg{}.win{}.sample{}.epochs{}.mincount{}.size{}.model'
        .format(os.path.basename(datadir), train_mode, alpha, neg, window_size,
                sample, epochs, min_count, size))


class Samples():  # pylint:disable=R0903
    """An iterable class (with generators) for gensim and n2v."""

    def __init__(self, source, shuffle, input_data=None):
        """Initialize instance."""
        logger.info('Loading {} samples...'.format(source))
        if source not in ['wiki', 'men', 'def', 'l2', 'l4', 'l6']:
            raise Exception('Invalid source parameter \'{}\''.format(source))
        if source == 'wiki' and not input_data:
            raise Exception('You need to specify the input data when parsing '
                            'a Wikipedia txt dump')
        self._source = source
        self._datafile = input_data
        self._shuffle = shuffle

    def _iterate_over_wiki(self):
        with open(self._datafile, 'rt', encoding='utf-8') as input_stream:
            for line in input_stream:
                yield line.strip().split()

    @classmethod
    def _iterate_over_men(cls):
        with open(DATASETS['men'], 'r', encoding='utf-8') as men_stream:
            for line in men_stream:
                line = line.rstrip('\n')
                items = line.split()
                yield (items[0], items[1]), float(items[2])

    def _iterate_over_definitions(self):
        with open(DATASETS['def'], 'rt', encoding='utf-8') as input_stream:
            if self._shuffle:
                logger.info('Iterating over test set in shuffled order')
                input_stream = list(input_stream)
                random.shuffle(input_stream)
            for line in input_stream:
                fields = line.rstrip('\n').split('\t')
                nonce = fields[0]
                sentence = fields[1].replace('___', nonce).split()
                probe = '{}_true'.format(nonce)
                yield [sentence], nonce, probe

    def _iterate_over_chimeras(self):
        with open(DATASETS[self._source], 'rt', encoding='utf-8') as istream:
            if self._shuffle:
                logger.info('Iterating over test set in shuffled order')
                istream = list(istream)
                random.shuffle(istream)
            for num, line in enumerate(istream):
                nonce = 'chimera_nonce_{}'.format(num+1)
                fields = line.rstrip('\n').split('\t')
                sentences = [[token if token != '___' else nonce for token in
                              sent.strip().split(' ')] for sent in
                             fields[1].split('@@')]
                probes = fields[2].split(',')
                responses = fields[3].split(',')
                yield sentences, nonce, probes, responses

    def __iter__(self):
        """Iterate over items with generators."""
        if self._source == 'wiki':
            return self._iterate_over_wiki()
        if self._source == 'men':
            return self._iterate_over_men()
        if self._source == 'def':
            return self._iterate_over_definitions()
        return self._iterate_over_chimeras()
