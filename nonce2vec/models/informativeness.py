"""Informativeness model.

Loads a bi-directional language model and computes various
entropy-based informativeness measures.
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
        self._sum_filter = sum_filter
        if sum_filter and sum_filter != 'random' and sum_thresh is None:
            raise Exception('Setting sum_filter as \'{}\' requires specifying '
                            'a threshold parameter'.format(sum_filter))
        self._sum_thresh = sum_thresh
        self._train_filter = train_filter
        if train_filter and train_filter != 'random' and train_thresh is None:
            raise Exception('Setting train_filter as \'{}\' requires specifying '
                            'a threshold parameter'.format(train_filter))
        self._train_thresh = train_thresh
        self._model = Word2Vec.load(model_path)
        self._sort_by = sort_by

    @property
    def sum_filter(self):
        return self._sum_filter

    @sum_filter.setter
    def sum_filter(self, sum_filter):
        self._sum_filter = sum_filter


    @lru_cache(maxsize=10)
    def _get_prob_distribution(self, context):
        words_and_probs = self._model.predict_output_word(
            context, topn=len(self._model.wv.vocab))
        return [item[1] for item in words_and_probs]

    @lru_cache(maxsize=10)
    def _get_context_entropy(self, context):
        if not context:
            return 0
        probs = self._get_prob_distribution(context)
        shannon_entropy = scipy.stats.entropy(probs)
        ctx_ent = 1 - (shannon_entropy / np.log(len(probs)))
        return ctx_ent

    @lru_cache(maxsize=50)
    def _get_context_word_entropy(self, context, word_index):
        ctx_ent_with_word = self._get_context_entropy(context)
        ctx_without_words = tuple(x for idx, x in enumerate(context) if
                                  idx != word_index)
        ctx_ent_without_word = self._get_context_entropy(ctx_without_words)
        cwe = ctx_ent_with_word - ctx_ent_without_word
        return cwe

    @lru_cache(maxsize=50)
    def _keep_item(self, idx, context, filter_type, threshold):
        if not filter_type:
            return True
        if filter_type == 'random':
            return self._model.wv.vocab[context[idx]].sample_int > self._model.random.rand() * 2 ** 32
        if filter_type == 'self':
            return np.log(self._model.wv.vocab[context[idx]].sample_int) > threshold
        if filter_type == 'cwe':
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

    def _get_in_vocab_context(self, sentence, vocab, nonce):
        return tuple([w for w in sentence if w in vocab and w != nonce])

    def _get_filtered_train_ctx_ent(self, sentences, vocab, nonce):
        ctx_ent = []
        for sentence in sentences:
            context = self._get_in_vocab_context(sentence, vocab, nonce)
            for idx, ctx in enumerate(context):
                if self._keep_item(idx, context, self._train_filter,
                                   self._train_thresh):
                    cwe = self._get_context_word_entropy(context, idx)
                    logger.debug('word = {} | cwe = {}'.format(context[idx],
                                                               cwe))
                    ctx_ent.append((ctx, cwe))
        return ctx_ent

    def filter_and_sort_train_ctx_ent(self, sentences, vocab, nonce):
        """Sort context and return a list of (ctx_word, ctx_word_entropy)."""
        logger.debug('Filtering and sorting train context...')
        ctx_ent = self._get_filtered_train_ctx_ent(sentences, vocab, nonce)
        if not self._sort_by:
            return ctx_ent
        if self._sort_by == 'desc':
            return sorted(ctx_ent, key=lambda x: x[1], reverse=True)
        if self._sort_by == 'asc':
            return sorted(ctx_ent, key=lambda x: x[1])
        raise Exception('Invalid sort_by value: {}'.format(self._sort_by))

    def filter_sum_context(self, sentences, vocab, nonce):
        """Filter the context to be summed over."""
        logger.debug('Filtering sum context...')
        filtered_ctx = []
        raw_ctx = []
        for sentence in sentences:
            _ctx = self._get_in_vocab_context(sentence, vocab, nonce)
            _filtered_ctx = self._filter_context(_ctx, self._sum_filter,
                                                 self._sum_thresh)
            raw_ctx.extend(list(_ctx))
            filtered_ctx.extend(list(_filtered_ctx))
        logger.debug('Filtered sum context = {}'.format(filtered_ctx))
        return raw_ctx, filtered_ctx
