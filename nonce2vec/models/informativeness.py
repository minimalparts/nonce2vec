"""Informativeness model.

Loads a language model and computes various entropy-based informativeness
measures.
"""

from functools import lru_cache

import logging

import numpy as np
import scipy

from gensim.models import Word2Vec


__all__ = ('Informativeness')

logger = logging.getLogger(__name__)


class Informativeness():
    """Informativeness class relying on a bi-directional language model."""

    def __init__(self, model_path, sum_filter=None, sum_thresh=None,
                 train_filter=None, train_thresh=None, sort_by=None):
        """Initialize the Informativeness instance.

        Args:
            model_path (str): The absolute path to the gensim w2v CBOW model.
            sum_filter (str): Filter for the sum initialization phase.
            sum_thresh (int): Threshold for sum filter (self and cwi filters
                              only).
            train_filter (str): Filter for the training phase.
            train_thresh (int): Threshold for the train filter (self and cwi
                                filters only).
            sort_by (str): Sort context items in asc or desc of cwi values
                           before training.
        """
        self._sum_filter = sum_filter
        if sum_filter and sum_filter != 'random' and sum_thresh is None:
            raise Exception('Setting sum_filter as \'{}\' requires specifying '
                            'a threshold parameter'.format(sum_filter))
        self._sum_thresh = sum_thresh
        self._train_filter = train_filter
        if train_filter and train_filter != 'random' and train_thresh is None:
            raise Exception('Setting train_filter as \'{}\' requires '
                            'specifying a threshold parameter'
                            .format(train_filter))
        self._train_thresh = train_thresh
        self._model = Word2Vec.load(model_path)
        self._sort_by = sort_by

    @property
    def sum_filter(self):
        """Return sum filter attribute."""
        return self._sum_filter

    @sum_filter.setter
    def sum_filter(self, sum_filter):
        self._sum_filter = sum_filter

    @lru_cache(maxsize=10)
    def _get_prob_distribution(self, context):
        words_and_probs = self._model.predict_output_word(
            context, topn=len(self._model.wv))
        return [item[1] for item in words_and_probs]

    @lru_cache(maxsize=10)
    def _get_context_entropy(self, context):
        if not context:
            return 0
        probs = self._get_prob_distribution(context)
        if not probs:
            return 0
        shannon_entropy = scipy.stats.entropy(probs)
        ctx_ent = 1 - (shannon_entropy / np.log(len(probs)))
        return ctx_ent

    @lru_cache(maxsize=50)
    def _get_context_word_entropy(self, context, word_index):
        ctx_ent_with_word = self._get_context_entropy(context)
        ctx_without_word = tuple(x for idx, x in enumerate(context) if
                                 idx != word_index)
        ctx_ent_without_word = self._get_context_entropy(ctx_without_word)
        cwi = ctx_ent_with_word - ctx_ent_without_word
        return cwi

    @lru_cache(maxsize=50)
    def _keep_item(self, idx, context, filter_type, threshold):
        if not filter_type:
            return True
        if filter_type == 'random':
            return self._model.wv.get_vecattr(context[idx], "sample_int") \
             > self._model.random.rand() * 2 ** 32
        if filter_type == 'self':
            return np.log(self._model.wv.get_vecattr(context[idx], "sample_int")) \
             > threshold
        if filter_type == 'cwi':
            return self._get_context_word_entropy(context, idx) > threshold
        raise Exception('Invalid ctx_filter parameter: {}'.format(filter_type))

    def _filter_context(self, context, filter_type, threshold):
        if not filter_type:
            logger.warning('Applying no filters to context selection: '
                           'this should negatively, and significantly, '
                           'impact results')
        else:
            logger.debug('Filtering with filter: {} and threshold = {}'
                         .format(filter_type, threshold))
        return tuple(ctx for idx, ctx in enumerate(context) if
                     self._keep_item(idx, context, filter_type, threshold))

    @classmethod
    def _get_in_vocab_context(cls, sentence, keyed_vectors, nonce):
        return tuple([w for w in sentence if w in keyed_vectors and w != nonce])

    def get_ctx_ent_for_weighted_sum(self, sentences, keyed_vectors, nonce):
        """Return context entropy."""
        ctx_ent_map = {}
        ctx_ent = self._get_filtered_train_ctx_ent(sentences, keyed_vectors, nonce)
        for ctx, cwi in ctx_ent:
            if ctx not in ctx_ent_map:
                ctx_ent_map[ctx] = cwi
            else:
                if cwi > ctx_ent_map[ctx]:
                    ctx_ent_map[ctx] = cwi
        return ctx_ent_map

    def _get_filtered_train_ctx_ent(self, sentences, keyed_vectors, nonce):
        ctx_ent = []
        for sentence in sentences:
            context = self._get_in_vocab_context(sentence, keyed_vectors, nonce)
            for idx, ctx in enumerate(context):
                if self._keep_item(idx, context, self._train_filter,
                                   self._train_thresh):
                    cwi = self._get_context_word_entropy(context, idx)
                    logger.debug('word = {} | cwi = {}'.format(context[idx],
                                                               cwi))
                    ctx_ent.append((ctx, cwi))
        return ctx_ent

    def filter_and_sort_train_ctx_ent(self, sentences, keyed_vectors, nonce):
        """Sort context and return a list of (ctx_word, ctx_word_entropy)."""
        logger.debug('Filtering and sorting train context...')
        ctx_ent = self._get_filtered_train_ctx_ent(sentences, keyed_vectors, nonce)
        if not self._sort_by:
            return ctx_ent
        if self._sort_by == 'desc':
            return sorted(ctx_ent, key=lambda x: x[1], reverse=True)
        if self._sort_by == 'asc':
            return sorted(ctx_ent, key=lambda x: x[1])
        raise Exception('Invalid sort_by value: {}'.format(self._sort_by))

    def filter_sum_context(self, sentences, keyed_vectors, nonce):
        """Filter the context to be summed over."""
        logger.debug('Filtering sum context...')
        filtered_ctx = []
        raw_ctx = []
        for sentence in sentences:
            _ctx = self._get_in_vocab_context(sentence, keyed_vectors, nonce)
            _filtered_ctx = self._filter_context(_ctx, self._sum_filter,
                                                 self._sum_thresh)
            raw_ctx.extend(list(_ctx))
            filtered_ctx.extend(list(_filtered_ctx))
        logger.debug('Filtered sum context = {}'.format(filtered_ctx))
        return raw_ctx, filtered_ctx
