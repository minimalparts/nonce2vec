# -*- encoding: utf-8 -*-
# pylint: skip-file
"""Nonce2Vec model.

A modified version of gensim.Word2Vec.
"""

import logging
from collections import defaultdict, OrderedDict
from typing import Optional

import numpy as np
from scipy.special import expit
from six import iteritems
from six.moves import xrange
from gensim.models.word2vec import Word2Vec, Word2VecVocab, Word2VecTrainables
from gensim.utils import keep_vocab_item


__all__ = ('Nonce2Vec')

from nonce2vec.models.informativeness import Informativeness

logger = logging.getLogger(__name__)


def compute_cwi_alpha(cwi, kappa, beta, alpha, min_alpha):
    x = np.tanh(cwi*beta)
    decay = (np.exp(kappa*(x+1)) - 1) / (np.exp(2*kappa) - 1)
    if decay * alpha > min_alpha:
        return decay * alpha
    return min_alpha


def compute_exp_alpha(nonce_count, lambda_den, alpha, min_alpha):
    exp_decay = -(nonce_count-1) / lambda_den
    if alpha * np.exp(exp_decay) > min_alpha:
        return alpha * np.exp(exp_decay)
    return min_alpha


def train_sg_pair_replication(model, word, context_index, alpha,
                              nonce_count, learn_vectors=True,
                              learn_hidden=True, context_vectors=None,
                              context_locks=None, compute_loss=False,
                              is_ft=False):
    if context_vectors is None:
        # context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors

    if context_locks is None:
        # context_locks = model.syn0_lockf
        context_locks = model.vectors_lockf

    if word not in model.wv:
        return
    predict_word_idx: int = model.wv.key_to_index[word]

    l1 = context_vectors[context_index]
    neu1e = np.zeros(l1.shape)

    # Only train the nonce
    if model.current_nonce is not None \
     and model.wv.index_to_key[context_index] == model.current_nonce \
     and word != model.current_nonce:
        lock_factor = context_locks[context_index]
        lambda_den = model.lambda_den
        exp_decay = -(nonce_count-1) / lambda_den
        if alpha * np.exp(exp_decay) > model.min_alpha:
            alpha = alpha * np.exp(exp_decay)
        else:
            alpha = model.min_alpha
        logger.debug('training on \'{}\' and \'{}\' with '
                     'alpha = {}'.format(model.current_nonce,
                                         word,
                                         round(alpha, 5)))
        if model.negative:
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word_idx]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                if w != predict_word_idx:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = np.dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients
            # multiplied by the learning rate
            if learn_hidden:
                model.syn1neg[word_indices] += np.outer(gb, l1)
                # learn hidden -> output
            neu1e += np.dot(gb, l2b)  # save error

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden
                # (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e


def train_batch_sg_replication(model, sentences, alpha, work=None,
                               compute_loss=False):
    result = 0
    window = model.window
    # Count the number of times that we see the nonce
    nonce_count = 0
    for sentence in sentences:
        word_vocabs_idx = [model.wv.key_to_index[w] for w in sentence if w in
                       model.wv and model.wv.get_vecattr(w, "sample_int")
                       > model.random.rand() * 2 ** 32 or w == '___']
        for pos, word_idx in enumerate(word_vocabs_idx):
            # Note: we have got rid of the random window size
            start = max(0, pos - window)
            for pos2, word2_idx in enumerate(word_vocabs_idx[start:(pos + window + 1)],
                                             start):
                # don't train on the `word` itself
                if pos2 != pos:
                    # If training context nonce, increment its count
                    if model.wv.index_to_key[word2_idx] == \
                     model.current_nonce:
                        nonce_count += 1
                        train_sg_pair_replication(
                            model, model.wv.index_to_key[word_idx],
                            word2_idx, alpha, nonce_count,
                            compute_loss=compute_loss)

        result += len(word_vocabs_idx)
        if window - 1 >= 3:
            window = window - model.window_decay
        model.recompute_sample_ints()
    return result


def train_sg_pair(model, word, context_index, alpha,
                  learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None, compute_loss=False,
                  is_ft=False):
    if context_vectors is None:
        # context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors

    if context_locks is None:
        # context_locks = model.syn0_lockf
        context_locks = model.vectors_lockf

    if word not in model.wv:
        return
    predict_word_idx = model.wv.key_to_index[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    neu1e = np.zeros(l1.shape)

    # Only train the nonce
    if model.current_nonce is not None \
     and model.wv.index_to_key[context_index] == model.current_nonce \
     and word != model.current_nonce:
        lock_factor = context_locks[context_index]
        if model.negative:
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word_idx]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                if w != predict_word_idx:
                    word_indices.append(w)
            l2b = model.syn1neg[word_indices]  # 2d matrix, k+1 x layer1_size
            prod_term = np.dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients
            # multiplied by the learning rate
            if learn_hidden:
                model.syn1neg[word_indices] += np.outer(gb, l1)
                # learn hidden -> output
            neu1e += np.dot(gb, l2b)  # save error

        if learn_vectors:
            l1 += neu1e * lock_factor  # learn input -> hidden
                # (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e


def _get_unique_ctx_ent_tuples(ctx_ent_tuples):
    ctx_ent_dict = OrderedDict()
    for ctx, ent in ctx_ent_tuples:
        if ctx not in ctx_ent_dict:
            ctx_ent_dict[ctx] = ent
        else:
            ctx_ent_dict[ctx] = max(ent, ctx_ent_dict[ctx])
    return [(ctx, ent) for ctx, ent in ctx_ent_dict.items()]


def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
    result = 0
    alpha = model.alpha  # re-initialize learning rate before each batch
    ctx_ent_tuples = model.trainables_info.filter_and_sort_train_ctx_ent(
        sentences, model.wv, model.current_nonce)
    if model.train_over_set:
        logger.debug('Training over set of context items')
        ctx_ent_tuples = _get_unique_ctx_ent_tuples(ctx_ent_tuples)
    logger.debug('Training on context = {}'.format(ctx_ent_tuples))
    nonce_vocab_idx = model.wv.key_to_index[model.current_nonce]
    nonce_count = 0
    for ctx_word, cwi in ctx_ent_tuples:
        ctx_vocab_idx = model.wv.key_to_index[ctx_word]
        nonce_count += 1
        if not model.train_with:
            raise Exception('Unspecified learning rate decay function. '
                            'You must specify a \'train_with\' parameter')
        if model.train_with == 'cwi_alpha':
            alpha = compute_cwi_alpha(cwi, model.kappa, model.beta, model.alpha,
                                      model.min_alpha)
            logger.debug('training on \'{}\' and \'{}\' with cwi = {}, b_cwi = {}, '
                         'alpha = {}'.format(model.wv.index_to_key[nonce_vocab_idx],
                                             model.wv.index_to_key[ctx_vocab_idx],
                                             round(cwi, 5),
                                             round(np.tanh(model.beta * cwi), 4),
                                             round(alpha, 5)))
        if model.train_with == 'exp_alpha':
            alpha = compute_exp_alpha(nonce_count, model.lambda_den,
                                      model.alpha, model.min_alpha)
            logger.debug('training on \'{}\' and \'{}\' with cwi = {}, '
                         'alpha = {}'.format(model.wv.index_to_key[nonce_vocab_idx],
                                             model.wv.index_to_key[ctx_vocab_idx],
                                             round(cwi, 5),
                                             round(alpha, 5)))
        if model.train_with == 'cst_alpha':
            alpha = model.alpha
        train_sg_pair(model, model.wv.index_to_key[ctx_vocab_idx],
                      nonce_vocab_idx, alpha, compute_loss=compute_loss)
        result += len(ctx_ent_tuples) + 1
    return result


class Nonce2VecVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3,
                 sorted_vocab=True, null_word=0):
        super(Nonce2VecVocab, self).__init__(max_vocab_size, min_count, sample,
                                             sorted_vocab, null_word)
        self.nonce = None

    # @classmethod
    # def load(cls, w2v_vocab):
    #     """Load a Nonce2VecVocab instance from a Word2VecVocab instance."""
    #     n2v_vocab = cls()
    #     for key, value in w2v_vocab.__dict__.items():
    #         setattr(n2v_vocab, key, value)
    #     return n2v_vocab




class Nonce2VecTrainables(Word2VecTrainables):

    def __init__(self, vector_size=100, seed=1, hashfxn=hash):
        super(Nonce2VecTrainables, self).__init__(vector_size, seed, hashfxn)
        self.info = None

    # @classmethod
    # def load(cls, w2v_trainables):
    #     n2v_trainables = cls()
    #     for key, value in w2v_trainables.__dict__.items():
    #         setattr(n2v_trainables, key, value)
    #     return n2v_trainables


class Nonce2Vec(Word2Vec):

    MAX_WORDS_IN_BATCH = 10000

    def __init__(self, sentences=None, vector_size=100, alpha=0.025, window=5,
                 min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=1, hs=0, negative=5, ns_exponent=0.75,
                 cbow_mean=1, hashfxn=hash, epochs=5, null_word=0,
                 trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False,
                 callbacks=(), max_final_vocab=None, window_decay=0,
                 sample_decay=1.0):
        super(Nonce2Vec, self).__init__(sentences=sentences, corpus_file=None,
                                        vector_size=vector_size, alpha=alpha,
                                        window=window, min_count=min_count,
                                        max_vocab_size=max_vocab_size, sample=sample,
                                        seed=seed, workers=workers, min_alpha=min_alpha,
                                        sg=sg, hs=hs, negative=negative,
                                        ns_exponent=ns_exponent, cbow_mean=cbow_mean,
                                        hashfxn=hashfxn, epochs=epochs,
                                        null_word=null_word, trim_rule=trim_rule,
                                        sorted_vocab=sorted_vocab, batch_words=batch_words,
                                        compute_loss=compute_loss, callbacks=callbacks)
        # self.trainables = Nonce2VecTrainables(seed=seed, vector_size=vector_size,
        #                                       hashfxn=hashfxn)
        self.lambda_den = 0.0
        self.sample_decay = float(sample_decay)
        self.window_decay = int(window_decay)
        self.trainables_info: Optional[Informativeness] = None
        self.current_nonce: Optional[str] = None

    @classmethod
    def load(cls, *args, **kwargs):
        w2vec_model = super(Nonce2Vec, cls).load(*args, **kwargs)
        n2vec_model = cls()
        for key, value in w2vec_model.__dict__.items():
            setattr(n2vec_model, key, value)
        return n2vec_model

    def _do_train_job(self, sentences, alpha, inits):
        """Train a single batch of sentences.

        Return 2-tuple `(effective word count after ignoring unknown words
        and sentence length trimming, total word count)`.
        """
        work, neu1 = inits
        tally = 0
        if self.sg:
            if self.replication:
                logger.info('Training n2v with original code')
                tally += train_batch_sg_replication(self, sentences, alpha,
                                                    work)
            else:
                logger.info('Training n2v with refactored code')
                tally += train_batch_sg(self, sentences, alpha, work)
        else:
            raise Exception('Nonce2Vec does not support cbow mode')
        return tally, self._raw_word_count(sentences)

    def build_vocab(self, sentences, update=False, progress_per=10000,
                    keep_raw_vocab=False, trim_rule=None, **kwargs):
        total_words, corpus_count = self.scan_vocab(
            sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        report_values, pre_exist_words = self.prepare_vocab(
            self.hs, self.negative, self.wv, update=update,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(
            vocab_size=report_values['num_retained_words'])
        self.prepare_weights(pre_exist_words, self.hs,
                             self.negative, self.wv,
                             sentences, self.current_nonce,
                             update=update,
                             replication=self.replication,
                             sum_over_set=self.sum_over_set,
                             weighted=self.weighted, beta=self.beta)

    def recompute_sample_ints(self):
        for w in self.wv.key_to_index.keys():
            old_value = float(self.wv.get_vecattr(w, "sample_int"))
            self.wv.set_vecattr(w, "sample_int", int(round(old_value / float(self.sample_decay))))

    def prepare_weights(self, pre_exist_words, hs, negative, wv, sentences,
                        nonce, update=False, replication=False,
                        sum_over_set=False, weighted=False, beta=1000):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            raise Exception('prepare_weight on Nonce2VecTrainables should '
                            'always be used with update=True')
        else:
            self.update_weights(pre_exist_words, hs, negative, wv, sentences,
                                nonce, replication, sum_over_set, weighted,
                                beta)

    def update_weights(self, pre_exist_words, hs, negative, wv, sentences,
                       nonce, replication=False, sum_over_set=False,
                       weighted=False, beta=1000):
        """
        Copy all the existing weights, and reset the weights for the newly
        added vocabulary.
        """
        logger.info('updating layer weights')
        gained_vocab = len(wv) - len(wv.vectors)
        # newvectors = empty((gained_vocab, wv.vector_size), dtype=REAL)
        newvectors = np.zeros((gained_vocab, wv.vector_size), dtype=np.float32)

        # randomize the remaining words
        # FIXME as-is the code is bug-prone. We actually only want to
        # initialize the vector for the nonce, not for the remaining gained
        # vocab. This implies that the system should be run with the same
        # min_count as the pre-trained background model. Otherwise
        # we won't be able to sum as we won't have vectors for the other
        # gained background words
        if gained_vocab > 1:
            raise Exception('Creating sum vector for non-nonce word. Do '
                            'not specify a min_count when running Nonce2Vec.')
        if gained_vocab == 0:
            raise Exception('Nonce word \'{}\' already in test set and not '
                            'properly deleted'.format(nonce))
        for i in xrange(len(wv.vectors), len(wv)):
            # Initialise to sum
            raw_ctx, filtered_ctx = self.trainables_info.filter_sum_context(
                sentences, pre_exist_words, nonce)
            if sum_over_set or replication:
                raw_ctx = set(raw_ctx)
                filtered_ctx = set(filtered_ctx)
                logger.debug('Summing over set of context items: {}'
                             .format(filtered_ctx))
            if weighted:
                logger.debug('Applying weighted sum')  # Sum over positive cwi words only
                ctx_ent_map = self.trainables_info.get_ctx_ent_for_weighted_sum(
                    sentences, pre_exist_words, nonce)
            if filtered_ctx:
                for w in filtered_ctx:
                    # Initialise to sum
                    if weighted:
                        # hacky reuse of compute_cwi_alpha to compute the
                        # weighted sum with cwi but compensating with
                        # beta for narrow distrib of cwi
                        newvectors[i-len(wv.vectors)] += wv.vectors[
                            wv.key_to_index[w]] * compute_cwi_alpha(
                                ctx_ent_map[w], kappa=1, beta=beta, alpha=1,
                                min_alpha=0)
                    else:
                        newvectors[i-len(wv.vectors)] += wv.vectors[
                            wv.key_to_index[w]]
            # If no filtered word remains, sum over everything to get 'some'
            # information
            else:
                logger.warning(
                    'No words left to sum over given filter settings. '
                    'Backtracking to sum over all raw context words')
                for w in raw_ctx:
                    # Initialise to sum
                    newvectors[i-len(wv.vectors)] += wv.vectors[
                        wv.key_to_index[w]]

        # Raise an error if an online update is run before initial training on
        # a corpus
        if not len(wv.vectors):
            raise RuntimeError('You cannot do an online vocabulary-update of a '
                               'model which has no prior vocabulary. First '
                               'build the vocabulary of your model with a '
                               'corpus before doing an online update.')

        wv.vectors = np.vstack([wv.vectors, newvectors])
        if negative:
            self.syn1neg = np.vstack([self.syn1neg,
                                      np.zeros((gained_vocab,
                                                self.layer1_size),
                                               dtype=np.float32)])
        wv.vectors_norm = None

        # do not suppress learning for already learned words
        self.vectors_lockf = np.ones(len(wv),
                                     dtype=np.float32)

    def prepare_vocab(self, hs, negative, wv, update=False,
                      keep_raw_vocab=False, trim_rule=None,
                      min_count=None, sample=None, dry_run=False):
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            raise Exception('Nonce2Vec can only update a pre-existing '
                            'vocabulary')
        logger.info('Updating model with new vocabulary')
        new_total = pre_exist_total = 0
        # New words and pre-existing words are two separate lists
        new_words = []
        pre_exist_words = []
        if self.current_nonce is not None:
            if self.current_nonce in wv:
                gold_nonce = '{}_true'.format(self.current_nonce)
                nonce_index = wv.key_to_index[self.current_nonce]
                wv.key_to_index[gold_nonce] = wv.key_to_index[self.current_nonce]
                wv.index_to_key[nonce_index] = gold_nonce
                # del wv.index_to_key[wv.vocab[self.nonce].index]
                del wv.key_to_index[self.current_nonce]
            for word, v in iteritems(self.raw_vocab):
                # Update count of all words already in vocab
                if word in wv:
                    pre_exist_words.append(word)
                    pre_exist_total += v
                    if not dry_run:
                        wv.set_vecattr(word, "count", wv.get_vecattr(word, "count") + v )
                else:
                    # For new words, keep the ones above the min count
                    # AND the nonce (regardless of count)
                    if keep_vocab_item(word, v, min_count,
                                       trim_rule=trim_rule) or word == self.current_nonce:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            wv.key_to_index[word] = len(wv)
                            wv.index_to_key.append(word)
                            wv.set_vecattr(word, "count", v)
                    else:
                        drop_unique += 1
                        drop_total += v
            original_unique_total = len(pre_exist_words) \
                + len(new_words) + drop_unique
            pre_exist_unique_pct = len(pre_exist_words) \
                * 100 / max(original_unique_total, 1)
            new_unique_pct = len(new_words) * 100 / max(original_unique_total, 1)
            logger.info('New added %i unique words (%i%% of original %i) '
                        'and increased the count of %i pre-existing words '
                        '(%i%% of original %i)', len(new_words),
                        new_unique_pct, original_unique_total,
                        len(pre_exist_words), pre_exist_unique_pct,
                        original_unique_total)
            retain_words = new_words + pre_exist_words
            retain_total = new_total + pre_exist_total

        # Precalculate each vocabulary item's threshold for sampling
        if not sample:
            # no words downsampled
            threshold_count = retain_total
        # Only retaining one subsampling notion from original gensim implementation
        else:
            threshold_count = sample * retain_total

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = wv.get_vecattr(w, "count")
            word_probability = (np.sqrt(v / threshold_count) + 1) \
                * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                wv.set_vecattr(w, "sample_int",  np.uintc(round(word_probability * 2**32)))

        if not dry_run and not keep_raw_vocab:
            logger.info('deleting the raw counts dictionary of %i items',
                        len(self.raw_vocab))
            self.raw_vocab = defaultdict(int)

        logger.info('sample=%g downsamples %i most-common words', sample,
                    downsample_unique)
        logger.info('downsampling leaves estimated %i word corpus '
                    '(%.1f%% of prior %i)', downsample_total,
                    downsample_total * 100.0 / max(retain_total, 1),
                    retain_total)

        # return from each step: words-affected, resulting-corpus-size,
        # extra memory estimates
        report_values = {
            'drop_unique': drop_unique, 'retain_total': retain_total,
            'downsample_unique': downsample_unique,
            'downsample_total': int(downsample_total),
            'num_retained_words': len(retain_words)
        }

        if self.null_word:
            # create null pseudo-word for padding when using concatenative
            # L1 (run-of-words)
            # this word is only ever input – never predicted – so count,
            # huffman-point, etc doesn't matter
            self.add_null_word(wv)

        if self.sorted_vocab and not update:
            self.sort_vocab(wv)
        if hs:
            # add info about each word's Huffman encoding
            self.create_binary_tree(wv)
        if negative:
            # build the table for drawing random words (for negative sampling)
            self.make_cum_table()

        return report_values, pre_exist_words
