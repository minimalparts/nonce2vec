"""Context filter."""

import logging

import numpy as np

__all__ = ('NoFilter', 'RandomFilter', 'SelfInformationFilter',
           'ContextWordEntropyFilter')

logger = logging.getLogger(__name__)


class NoFilter():
    """No filter."""

    def filter_tokens(self, tokens):
        return tokens


class RandomFilter():
    """A random filter using w2v subsampling."""
    def __init__(self, model):
        self._model = model  # a w2v model

    def filter_tokens(self, tokens):
        return [w for w in tokens if self._model.wv.vocab[w].sample_int >
                self._model.random.rand() * 2 ** 32
                or w == self._model.vocabulary.nonce]

class SelfInformationFilter():
    """A filter based on self information computed from subsampling metrics."""
    def __init__(self, model, threshold):
        self._model = model  # a w2v model
        self._threshold = threshold

    def filter_tokens(self, tokens):
        return [w for w in tokens if
                np.log(self._model.wv.vocab[w].sample_int)
                > self._threshold or w == self._model.vocabulary.nonce]

class ContextWordEntropyFilter():
    """A filter based on context word entropy."""
    def __init__(self, informativeness, threshold):
        self._entropy = {}
        self._info = informativeness
        self._threshold = threshold

    def filter_tokens(self, tokens):
        return [w for w in tokens if self._entropy[w] > self._threshold]

    def compute_entropy(self, sentences, nonce):
        """Set context word entropy values for a list of sentences."""
        logger.info('Computing context word entropy for all context words...')
        self._entropy = {}
        # FIXME: this will work with CBOW in general and BIDIR as long as
        # context contains one instance of each word. Overall it's the case in
        # all test sets but still, it's dirty. Final version should support
        # duplicate tokens and get informativeness based on their position
        for tokens in sentences:
            for idx, token in enumerate(tokens):
                if token != nonce:
                    self._entropy[token] = self._info.get_context_word_entropy(
                        tokens, idx, tokens.index(nonce))
                    print('token = {} | cwe = {}'.format(token, self._entropy[token]))
