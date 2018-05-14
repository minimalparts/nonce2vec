"""Informativeness model.

Loads a bi-directional language model and computes various
entropy-based informativeness measures.
"""

import copy

import logging

import numpy as np
import scipy

from gensim.models import Word2Vec


__all__ = ('Informativeness')

logger = logging.getLogger(__name__)


class Informativeness():
    """Informativeness class relying on a bi-directional language model."""

    def __init__(self, mode, model_path, ctx_filter=None, threshold=None,
                 entropy=None, stats=False):
        """Initialize the Informativeness instance.

        Args:
            torch_model_path (str): The absolute path to the pickled
                                    bdlm torch model.
            vocab_path (str): The absolute path to the .txt file storing the
                              bdlm model vocabulary.

        Returns:
            model: a bdlm.BDLM instance.
            vocab: a bdlm.Dictionary instance
            nlp: a spacy.nlp loaded instance (for English)
            cuda: set to True to use GPUs with pyTorch
        """
        self._mode = mode
        self._filter = ctx_filter
        if ctx_filter and ctx_filter != 'random' and threshold is None:
            raise Exception('Setting ctx_filter as \'{}\' requires specifying '
                            'a threshold parameter'.format(ctx_filter))
        self._threshold = threshold
        self._model = Word2Vec.load(model_path)
        self._entropy = entropy
        #self._stats = stats  # if set to True store computed info to analyze statistics (mean, std, etc.)
        self._ctx_ent_data = []
        self._cwe_data = []

    def filter_tokens(self, tokens, nonce):
        """Filter a list of tokens containing a nonce.

        Filter the list based on its informativeness towards the nonce.
        """
        logger.debug('Filtering tokens: {}'.format(tokens))
        if not self._filter:
            logger.warning('Applying no filters to context selection: '
                           'this should negatively, and significantly, '
                           'impact results')
            return tokens
        if self._filter == 'random':
            filtered_tokens = []
            for w in tokens:
                logger.debug('word = {} | sample_int = {}'
                             .format(w, self._model.wv.vocab[w].sample_int))
                if self._model.wv.vocab[w].sample_int > \
                self._model.random.rand() * 2 ** 32 or w == nonce:
                    filtered_tokens.append(w)
            logger.debug('Output filtered tokens = {}'.format(filtered_tokens))
            return filtered_tokens
        if self._filter == 'self':
            filtered_tokens = []
            for w in tokens:
                logger.debug('word = {} | log(sample_int) = {}'.format(
                    w, np.log(self._model.wv.vocab[w].sample_int)))
                if np.log(self._model.wv.vocab[w].sample_int) > \
                self._threshold or w == nonce:
                    filtered_tokens.append(w)
            logger.debug('Output filtered tokens = {}'.format(filtered_tokens))
            return filtered_tokens
        if self._filter == 'cwe':
            filtered_tokens = []
            for idx, w in enumerate(tokens):
                cwe = self.get_context_word_entropy(tokens, idx)
                logger.debug('word = {} | cwe = {}'.format(w, cwe))
                if cwe > self._threshold or w == nonce:
                    filtered_tokens.append(w)
            logger.debug('Output filtered tokens = {}'.format(filtered_tokens))
            return filtered_tokens
        raise Exception('Invalid ctx_filter parameter: {}'
                        .format(self._filter))

    def get_context_entropy(self, tokens):
        words_and_probs = self._model.predict_output_word(
            tokens, topn=len(self._model.wv.vocab))
        probs = [item[1] for item in words_and_probs]
        shannon_entropy = scipy.stats.entropy(probs)
        ctx_ent = 1 - (shannon_entropy / np.log(len(probs)))
        return ctx_ent

    def get_context_word_entropy(self, tokens, word_index):
        """Get how much a given word impacts the context entropy of a list of tokens.

        Characterizes how informative a source word is towards a target word
        in a given sequence of tokens.
        """
        ctx_ent_with_word = self.get_context_entropy(tokens)
        _tokens = copy.deepcopy(tokens)
        del _tokens[word_index]
        ctx_ent_without_word = self.get_context_entropy(_tokens)
        cwe = ctx_ent_with_word - ctx_ent_without_word
        # if self._stats:
        #     self._cwe_data.append(cwe)
        return cwe
