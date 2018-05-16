# -*- encoding: utf-8 -*-
"""Nonce2Vec model.

A modified version of gensim.Word2Vec.
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.special import expit
from six import iteritems
from six.moves import xrange
from gensim.models.word2vec import Word2Vec, Word2VecVocab, Word2VecTrainables
from gensim.utils import keep_vocab_item
from gensim.models.keyedvectors import Vocab

__all__ = ('Nonce2Vec')

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    print(compute_cwe_alpha())

def compute_cwe_alpha(x, k, b, alpha, min_alpha):
    x = np.tanh(x*b)
    decay = (np.exp(k*(x+1)) - 1) / (np.exp(2*k) - 1)
    if decay * alpha > min_alpha:
        return decay * alpha
    return min_alpha

def compute_exp_alpha(nonce_count, lambda_den, alpha, min_alpha):
    exp_decay = -(nonce_count-1) / lambda_den
    if alpha * np.exp(exp_decay) > min_alpha:
        return alpha * np.exp(exp_decay)
    return min_alpha

def train_sg_pair(model, word, context_index, alpha,
                  learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None, compute_loss=False,
                  is_ft=False):
    if context_vectors is None:
        #context_vectors = model.wv.syn0
        context_vectors = model.wv.vectors

    if context_locks is None:
        #context_locks = model.syn0_lockf
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
    return neu1e, alpha


def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
    result = 0
    alpha = model.alpha  # re-initialize learning rate before each batch
    ctx_ent_tuples = model.trainables.info.filter_and_sort_train_ctx_ent(
        sentences, model.wv.vocab, model.vocabulary.nonce)
    logger.debug('Training on context = {}'.format(ctx_ent_tuples))
    nonce_vocab = model.wv.vocab[model.vocabulary.nonce]
    nonce_count = 0
    for ctx_word, cwe in ctx_ent_tuples:
        ctx_vocab = model.wv.vocab[ctx_word]
        if not model.train_with:
            raise Exception('Unspecified learning rate decay function. '
                            'You must specify a \'train_with\' parameter')
        if model.train_with == 'exp_alpha':
            alpha = compute_exp_alpha(nonce_count, model.lambda_den, alpha,
                                      model.min_alpha)
            logger.debug('training on \'{}\' and \'{}\' with cwe = {}, '
                         'alpha = {}'.format(model.wv.index2word[nonce_vocab.index],
                                             model.wv.index2word[ctx_vocab.index],
                                             round(cwe, 5),
                                             round(alpha, 5)))
        if model.train_with == 'cwe_alpha':
            alpha = compute_cwe_alpha(cwe, model.k, model.bias, model.alpha,
                                      model.min_alpha)
            logger.debug('training on \'{}\' and \'{}\' with cwe = {}, b_cwe = {}, '
                         'alpha = {}'.format(model.wv.index2word[nonce_vocab.index],
                                             model.wv.index2word[ctx_vocab.index],
                                             round(cwe, 5),
                                             round(np.tanh(model.bias * cwe), 4),
                                             round(alpha, 5)))
        if model.train_with == 'cst_alpha':
            alpha = model.alpha

        _, alpha = train_sg_pair(model, model.wv.index2word[ctx_vocab.index],
                                 nonce_vocab.index, alpha,
                                 compute_loss=compute_loss)
        nonce_count += 1
        result += len(ctx_ent_tuples) + 1
    return result


class Nonce2VecVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3,
                 sorted_vocab=True, null_word=0):
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
        # If nonce is already in previous vocab, replace its label
        # (copy the original to a new slot, and delete original)
        if self.nonce is not None and self.nonce in wv.vocab:
            gold_nonce = '{}_true'.format(self.nonce)
            nonce_index = wv.vocab[self.nonce].index
            wv.vocab[gold_nonce] = wv.vocab[self.nonce]
            wv.index2word[nonce_index] = gold_nonce
            #del wv.index2word[wv.vocab[self.nonce].index]
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
        super(Nonce2VecTrainables, self).__init__(vector_size, seed, hashfxn)
        self.info = None

    @classmethod
    def load(cls, w2v_trainables):
        n2v_trainables = cls()
        for key, value in w2v_trainables.__dict__.items():
            setattr(n2v_trainables, key, value)
        return n2v_trainables

    def prepare_weights(self, pre_exist_words, hs, negative, wv, sentences,
                        nonce, update=False):
        """Build tables and model weights based on final vocabulary settings."""
        # set initial input/projection and hidden weights
        if not update:
            raise Exception('prepare_weight on Nonce2VecTrainables should '
                            'always be used with update=True')
        else:
            self.update_weights(pre_exist_words, hs, negative, wv, sentences,
                                nonce)

    def update_weights(self, pre_exist_words, hs, negative, wv, sentences,
                       nonce):
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
            if filtered_ctx:
                for w in filtered_ctx:
                    # Initialise to sum
                    newvectors[i-len(wv.vectors)] += wv.vectors[
                        wv.vocab[w].index]
            # If no filtered word remains, sum over everything to get 'some'
            # information
            else:
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
                                        dtype=np.float32)  # zeros suppress learning


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
            tally += train_batch_sg(self, sentences, alpha, work)
        else:
            raise Exception('Nonce2Vec does not support cbow mode')
        return tally, self._raw_word_count(sentences)

    def build_vocab(self, sentences, update=False, progress_per=10000,
                    keep_raw_vocab=False, trim_rule=None, **kwargs):
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
                                        update=update)

    def recompute_sample_ints(self):
        for w, o in self.wv.vocab.items():
            o.sample_int = int(round(float(o.sample_int) / float(self.sample_decay)))
