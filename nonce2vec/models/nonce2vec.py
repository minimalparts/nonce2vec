# -*- encoding: utf-8 -*-
# pylint: skip-file
"""Nonce2Vec model.

A modified version of gensim.Word2Vec.
"""

import logging
from collections import defaultdict, OrderedDict

import numpy as np
from scipy.special import expit
from six import iteritems
from six.moves import xrange
from gensim.models.word2vec import Word2Vec, Word2VecVocab, Word2VecTrainables
from gensim.utils import keep_vocab_item
from gensim.models.keyedvectors import Vocab

__all__ = ('Nonce2Vec')

logger = logging.getLogger(__name__)


def compute_cwi_alpha(cwi, kappa, beta, alpha, min_alpha):
    """
    R computes the alpha.

    Args:
        cwi: (todo): write your description
        kappa: (todo): write your description
        beta: (float): write your description
        alpha: (float): write your description
        min_alpha: (float): write your description
    """
    x = np.tanh(cwi*beta)
    decay = (np.exp(kappa*(x+1)) - 1) / (np.exp(2*kappa) - 1)
    if decay * alpha > min_alpha:
        return decay * alpha
    return min_alpha


def compute_exp_alpha(nonce_count, lambda_den, alpha, min_alpha):
    """
    Compute the exp_count ).

    Args:
        nonce_count: (todo): write your description
        lambda_den: (float): write your description
        alpha: (float): write your description
        min_alpha: (float): write your description
    """
    exp_decay = -(nonce_count-1) / lambda_den
    if alpha * np.exp(exp_decay) > min_alpha:
        return alpha * np.exp(exp_decay)
    return min_alpha


def train_sg_pair_replication(model, word, context_index, alpha,
                              nonce_count, learn_vectors=True,
                              learn_hidden=True, context_vectors=None,
                              context_locks=None, compute_loss=False,
                              is_ft=False):
    """
    Train a model.

    Args:
        model: (todo): write your description
        word: (str): write your description
        context_index: (str): write your description
        alpha: (float): write your description
        nonce_count: (todo): write your description
        learn_vectors: (bool): write your description
        learn_hidden: (int): write your description
        context_vectors: (str): write your description
        context_locks: (todo): write your description
        compute_loss: (bool): write your description
        is_ft: (bool): write your description
    """
    if context_vectors is None:
        # context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors

    if context_locks is None:
        # context_locks = model.syn0_lockf
        context_locks = model.trainables.vectors_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]

    l1 = context_vectors[context_index]
    neu1e = np.zeros(l1.shape)

    # Only train the nonce
    if model.vocabulary.nonce is not None \
     and model.wv.index2word[context_index] == model.vocabulary.nonce \
     and word != model.vocabulary.nonce:
        lock_factor = context_locks[context_index]
        lambda_den = model.lambda_den
        exp_decay = -(nonce_count-1) / lambda_den
        if alpha * np.exp(exp_decay) > model.min_alpha:
            alpha = alpha * np.exp(exp_decay)
        else:
            alpha = model.min_alpha
        logger.debug('training on \'{}\' and \'{}\' with '
                     'alpha = {}'.format(model.vocabulary.nonce,
                                         word,
                                         round(alpha, 5)))
        if model.negative:
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
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
    """
    Train a batch of sentences.

    Args:
        model: (todo): write your description
        sentences: (todo): write your description
        alpha: (float): write your description
        work: (int): write your description
        compute_loss: (bool): write your description
    """
    result = 0
    window = model.window
    # Count the number of times that we see the nonce
    nonce_count = 0
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in
                       model.wv.vocab and model.wv.vocab[w].sample_int
                       > model.random.rand() * 2 ** 32 or w == '___']
        for pos, word in enumerate(word_vocabs):
            # Note: we have got rid of the random window size
            start = max(0, pos - window)
            for pos2, word2 in enumerate(word_vocabs[start:(pos + window + 1)],
                                         start):
                # don't train on the `word` itself
                if pos2 != pos:
                    # If training context nonce, increment its count
                    if model.wv.index2word[word2.index] == \
                     model.vocabulary.nonce:
                        nonce_count += 1
                        train_sg_pair_replication(
                            model, model.wv.index2word[word.index],
                            word2.index, alpha, nonce_count,
                            compute_loss=compute_loss)

        result += len(word_vocabs)
        if window - 1 >= 3:
            window = window - model.window_decay
        model.recompute_sample_ints()
    return result


def train_sg_pair(model, word, context_index, alpha,
                  learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None, compute_loss=False,
                  is_ft=False):
    """
    Train pairwise model.

    Args:
        model: (todo): write your description
        word: (str): write your description
        context_index: (str): write your description
        alpha: (float): write your description
        learn_vectors: (bool): write your description
        learn_hidden: (int): write your description
        context_vectors: (todo): write your description
        context_locks: (todo): write your description
        compute_loss: (bool): write your description
        is_ft: (bool): write your description
    """
    if context_vectors is None:
        # context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors

    if context_locks is None:
        # context_locks = model.syn0_lockf
        context_locks = model.trainables.vectors_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    l1 = context_vectors[context_index]  # input word (NN input/projection layer)
    neu1e = np.zeros(l1.shape)

    # Only train the nonce
    if model.vocabulary.nonce is not None \
     and model.wv.index2word[context_index] == model.vocabulary.nonce \
     and word != model.vocabulary.nonce:
        lock_factor = context_locks[context_index]
        if model.negative:
            # use this word (label = 1) + `negative` other random words not
            # from this sentence (label = 0)
            word_indices = [predict_word.index]
            while len(word_indices) < model.negative + 1:
                w = model.cum_table.searchsorted(
                    model.random.randint(model.cum_table[-1]))
                if w != predict_word.index:
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
    """
    Returns a dictionary of unique entries.

    Args:
        ctx_ent_tuples: (str): write your description
    """
    ctx_ent_dict = OrderedDict()
    for ctx, ent in ctx_ent_tuples:
        if ctx not in ctx_ent_dict:
            ctx_ent_dict[ctx] = ent
        else:
            ctx_ent_dict[ctx] = max(ent, ctx_ent_dict[ctx])
    return [(ctx, ent) for ctx, ent in ctx_ent_dict.items()]


def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
    """
    Train model.

    Args:
        model: (todo): write your description
        sentences: (todo): write your description
        alpha: (float): write your description
        work: (int): write your description
        compute_loss: (bool): write your description
    """
    result = 0
    alpha = model.alpha  # re-initialize learning rate before each batch
    ctx_ent_tuples = model.trainables.info.filter_and_sort_train_ctx_ent(
        sentences, model.wv.vocab, model.vocabulary.nonce)
    if model.train_over_set:
        logger.debug('Training over set of context items')
        ctx_ent_tuples = _get_unique_ctx_ent_tuples(ctx_ent_tuples)
    logger.debug('Training on context = {}'.format(ctx_ent_tuples))
    nonce_vocab = model.wv.vocab[model.vocabulary.nonce]
    nonce_count = 0
    for ctx_word, cwi in ctx_ent_tuples:
        ctx_vocab = model.wv.vocab[ctx_word]
        nonce_count += 1
        if not model.train_with:
            raise Exception('Unspecified learning rate decay function. '
                            'You must specify a \'train_with\' parameter')
        if model.train_with == 'cwi_alpha':
            alpha = compute_cwi_alpha(cwi, model.kappa, model.beta, model.alpha,
                                      model.min_alpha)
            logger.debug('training on \'{}\' and \'{}\' with cwi = {}, b_cwi = {}, '
                         'alpha = {}'.format(model.wv.index2word[nonce_vocab.index],
                                             model.wv.index2word[ctx_vocab.index],
                                             round(cwi, 5),
                                             round(np.tanh(model.beta * cwi), 4),
                                             round(alpha, 5)))
        if model.train_with == 'exp_alpha':
            alpha = compute_exp_alpha(nonce_count, model.lambda_den,
                                      model.alpha, model.min_alpha)
            logger.debug('training on \'{}\' and \'{}\' with cwi = {}, '
                         'alpha = {}'.format(model.wv.index2word[nonce_vocab.index],
                                             model.wv.index2word[ctx_vocab.index],
                                             round(cwi, 5),
                                             round(alpha, 5)))
        if model.train_with == 'cst_alpha':
            alpha = model.alpha
        train_sg_pair(model, model.wv.index2word[ctx_vocab.index],
                      nonce_vocab.index, alpha, compute_loss=compute_loss)
        result += len(ctx_ent_tuples) + 1
    return result


class Nonce2VecVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3,
                 sorted_vocab=True, null_word=0):
        """
        Initialize vocab.

        Args:
            self: (todo): write your description
            max_vocab_size: (int): write your description
            min_count: (int): write your description
            sample: (todo): write your description
            sorted_vocab: (todo): write your description
            null_word: (todo): write your description
        """
        super(Nonce2VecVocab, self).__init__(max_vocab_size, min_count, sample,
                                             sorted_vocab, null_word)
        self.nonce = None

    @classmethod
    def load(cls, w2v_vocab):
        """Load a Nonce2VecVocab instance from a Word2VecVocab instance."""
        n2v_vocab = cls()
        for key, value in w2v_vocab.__dict__.items():
            setattr(n2v_vocab, key, value)
        return n2v_vocab

    def prepare_vocab(self, hs, negative, wv, update=False,
                      keep_raw_vocab=False, trim_rule=None,
                      min_count=None, sample=None, dry_run=False):
        """
        Prepare vocab for vocab.

        Args:
            self: (todo): write your description
            hs: (todo): write your description
            negative: (todo): write your description
            wv: (todo): write your description
            update: (todo): write your description
            keep_raw_vocab: (bool): write your description
            trim_rule: (todo): write your description
            min_count: (int): write your description
            sample: (int): write your description
            dry_run: (todo): write your description
        """
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
        if self.nonce is not None:
        # if self.nonce is not None and self.nonce in wv.vocab:
            if self.nonce in wv.vocab:
                gold_nonce = '{}_true'.format(self.nonce)
                nonce_index = wv.vocab[self.nonce].index
                wv.vocab[gold_nonce] = wv.vocab[self.nonce]
                wv.index2word[nonce_index] = gold_nonce
                # del wv.index2word[wv.vocab[self.nonce].index]
                del wv.vocab[self.nonce]
            for word, v in iteritems(self.raw_vocab):
                # Update count of all words already in vocab
                if word in wv.vocab:
                    pre_exist_words.append(word)
                    pre_exist_total += v
                    if not dry_run:
                        wv.vocab[word].count += v
                else:
                    # For new words, keep the ones above the min count
                    # AND the nonce (regardless of count)
                    if keep_vocab_item(word, v, min_count,
                                       trim_rule=trim_rule) or word == self.nonce:
                        new_words.append(word)
                        new_total += v
                        if not dry_run:
                            wv.vocab[word] = Vocab(count=v,
                                                   index=len(wv.index2word))
                            wv.index2word.append(word)
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
            v = wv.vocab[w].count
            word_probability = (np.sqrt(v / threshold_count) + 1) \
                * (threshold_count / v)
            if word_probability < 1.0:
                downsample_unique += 1
                downsample_total += word_probability * v
            else:
                word_probability = 1.0
                downsample_total += v
            if not dry_run:
                wv.vocab[w].sample_int = int(round(word_probability * 2**32))

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
            self.make_cum_table(wv)

        return report_values, pre_exist_words


class Nonce2VecTrainables(Word2VecTrainables):

    def __init__(self, vector_size=100, seed=1, hashfxn=hash):
        """
        Initialize hashfxables.

        Args:
            self: (todo): write your description
            vector_size: (int): write your description
            seed: (int): write your description
            hashfxn: (todo): write your description
            hash: (todo): write your description
        """
        super(Nonce2VecTrainables, self).__init__(vector_size, seed, hashfxn)
        self.info = None

    @classmethod
    def load(cls, w2v_trainables):
        """
        Loads all of - like object with the given w2v_trainables.

        Args:
            cls: (todo): write your description
            w2v_trainables: (str): write your description
        """
        n2v_trainables = cls()
        for key, value in w2v_trainables.__dict__.items():
            setattr(n2v_trainables, key, value)
        return n2v_trainables

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
        gained_vocab = len(wv.vocab) - len(wv.vectors)
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
        for i in xrange(len(wv.vectors), len(wv.vocab)):
            # Initialise to sum
            raw_ctx, filtered_ctx = self.info.filter_sum_context(
                sentences, pre_exist_words, nonce)
            if sum_over_set or replication:
                raw_ctx = set(raw_ctx)
                filtered_ctx = set(filtered_ctx)
                logger.debug('Summing over set of context items: {}'
                             .format(filtered_ctx))
            if weighted:
                logger.debug('Applying weighted sum')  # Sum over positive cwi words only
                ctx_ent_map = self.info.get_ctx_ent_for_weighted_sum(
                    sentences, pre_exist_words, nonce)
            if filtered_ctx:
                for w in filtered_ctx:
                    # Initialise to sum
                    if weighted:
                        # hacky reuse of compute_cwi_alpha to compute the
                        # weighted sum with cwi but compensating with
                        # beta for narrow distrib of cwi
                        newvectors[i-len(wv.vectors)] += wv.vectors[
                            wv.vocab[w].index] * compute_cwi_alpha(
                                ctx_ent_map[w], kappa=1, beta=beta, alpha=1,
                                min_alpha=0)
                    else:
                        newvectors[i-len(wv.vectors)] += wv.vectors[
                            wv.vocab[w].index]
            # If no filtered word remains, sum over everything to get 'some'
            # information
            else:
                logger.warning(
                    'No words left to sum over given filter settings. '
                    'Backtracking to sum over all raw context words')
                for w in raw_ctx:
                    # Initialise to sum
                    newvectors[i-len(wv.vectors)] += wv.vectors[
                        wv.vocab[w].index]

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
        self.vectors_lockf = np.ones(len(wv.vocab),
                                     dtype=np.float32)


class Nonce2Vec(Word2Vec):

    MAX_WORDS_IN_BATCH = 10000

    def __init__(self, sentences=None, size=100, alpha=0.025, window=5,
                 min_count=5, max_vocab_size=None, sample=1e-3, seed=1,
                 workers=3, min_alpha=0.0001, sg=1, hs=0, negative=5,
                 cbow_mean=1, hashfxn=hash, iter=5, null_word=0,
                 trim_rule=None, sorted_vocab=1,
                 batch_words=MAX_WORDS_IN_BATCH, compute_loss=False,
                 callbacks=(), max_final_vocab=None, window_decay=0,
                 sample_decay=1.0):
        """
        Initialize sentences.

        Args:
            self: (todo): write your description
            sentences: (str): write your description
            size: (int): write your description
            alpha: (float): write your description
            window: (int): write your description
            min_count: (int): write your description
            max_vocab_size: (int): write your description
            sample: (todo): write your description
            seed: (int): write your description
            workers: (int): write your description
            min_alpha: (float): write your description
            sg: (str): write your description
            hs: (int): write your description
            negative: (bool): write your description
            cbow_mean: (todo): write your description
            hashfxn: (todo): write your description
            hash: (todo): write your description
            iter: (todo): write your description
            null_word: (todo): write your description
            trim_rule: (int): write your description
            sorted_vocab: (todo): write your description
            batch_words: (todo): write your description
            MAX_WORDS_IN_BATCH: (int): write your description
            compute_loss: (bool): write your description
            callbacks: (list): write your description
            max_final_vocab: (int): write your description
            window_decay: (int): write your description
            sample_decay: (float): write your description
        """
        super(Nonce2Vec, self).__init__(sentences, size, alpha, window,
                                        min_count, max_vocab_size, sample,
                                        seed, workers, min_alpha, sg, hs,
                                        negative, cbow_mean, hashfxn, iter,
                                        null_word, trim_rule, sorted_vocab,
                                        batch_words, compute_loss, callbacks)
        self.trainables = Nonce2VecTrainables(seed=seed, vector_size=size,
                                              hashfxn=hashfxn)
        self.lambda_den = 0.0
        self.sample_decay = float(sample_decay)
        self.window_decay = int(window_decay)

    @classmethod
    def load(cls, *args, **kwargs):
        """
        Load a model from a n2vec.

        Args:
            cls: (todo): write your description
        """
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
        """
        Builds the vocab

        Args:
            self: (todo): write your description
            sentences: (todo): write your description
            update: (todo): write your description
            progress_per: (todo): write your description
            keep_raw_vocab: (bool): write your description
            trim_rule: (todo): write your description
        """
        total_words, corpus_count = self.vocabulary.scan_vocab(
            sentences, progress_per=progress_per, trim_rule=trim_rule)
        self.corpus_count = corpus_count
        report_values, pre_exist_words = self.vocabulary.prepare_vocab(
            self.hs, self.negative, self.wv, update=update,
            keep_raw_vocab=keep_raw_vocab, trim_rule=trim_rule, **kwargs)
        report_values['memory'] = self.estimate_memory(
            vocab_size=report_values['num_retained_words'])
        self.trainables.prepare_weights(pre_exist_words, self.hs,
                                        self.negative, self.wv,
                                        sentences, self.vocabulary.nonce,
                                        update=update,
                                        replication=self.replication,
                                        sum_over_set=self.sum_over_set,
                                        weighted=self.weighted, beta=self.beta)

    def recompute_sample_ints(self):
        """
        Re - calculate the number of samples.

        Args:
            self: (todo): write your description
        """
        for w, o in self.wv.vocab.items():
            o.sample_int = int(round(float(o.sample_int) / float(self.sample_decay)))
