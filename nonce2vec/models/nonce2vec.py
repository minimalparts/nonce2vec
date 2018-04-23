"""Nonce2Vec model.

A modified version of gensim.Word2Vec.
"""

import copy
import logging
from collections import defaultdict

import numpy
from scipy.special import expit
from six import iteritems
from gensim.models.word2vec import Word2Vec, Word2VecVocab
from gensim.utils import keep_vocab_item
from gensim.models.keyedvectors import Vocab

__all__ = ('Nonce2Vec')

logger = logging.getLogger(__name__)


def train_sg_pair(model, word, context_index, alpha,
                  nonce_count, learn_vectors=True, learn_hidden=True,
                  context_vectors=None, context_locks=None, compute_loss=False,
                  is_ft=False):
    if context_vectors is None:
        if is_ft:
            context_vectors_vocab = model.wv.syn0_vocab
            context_vectors_ngrams = model.wv.syn0_ngrams
        else:
            context_vectors = model.wv.syn0
    if context_locks is None:
        if is_ft:
            context_locks_vocab = model.syn0_vocab_lockf
            context_locks_ngrams = model.syn0_ngrams_lockf
        else:
            context_locks = model.syn0_lockf

    if word not in model.wv.vocab:
        return
    predict_word = model.wv.vocab[word]  # target word (NN output)

    if is_ft:
        l1_vocab = context_vectors_vocab[context_index[0]]
        l1_ngrams = numpy.np_sum(context_vectors_ngrams[context_index[1:]], axis=0)
        if context_index:
            l1 = numpy.np_sum([l1_vocab, l1_ngrams], axis=0) / len(context_index)
    else:
        l1 = context_vectors[context_index]  # input word (NN input/projection layer)
        lock_factor = context_locks[context_index]

    neu1e = numpy.zeros(l1.shape)

    # Only train the nonce
    if model.vocabulary.nonce is not None \
     and model.wv.index2word[context_index] == model.vocabulary.nonce \
     and word != model.vocabulary.nonce:
        lock_factor = context_locks[context_index]
        lambda_den = model.lambda_den
        exp_decay = -(nonce_count-1) / lambda_den
        if alpha * numpy.exp(exp_decay) > model.min_alpha:
            alpha = alpha * numpy.exp(exp_decay)
        else:
            alpha = model.min_alpha
        if model.hs:
            # work on the entire tree at once, to push as much work into
            # numpy's C routines as possible (performance)
            l2a = copy.deepcopy(model.syn1[predict_word.point])
            # 2d matrix, codelen x layer1_size
            prod_term = numpy.dot(l1, l2a.T)
            fa = expit(prod_term)  # propagate hidden -> output
            ga = (1 - predict_word.code - fa) * alpha  # vector of error
            # gradients multiplied by the learning rate
            if learn_hidden:
                model.syn1[predict_word.point] += numpy.outer(ga, l1)
                # learn hidden -> output
            neu1e += numpy.dot(ga, l2a)  # save error

            # loss component corresponding to hierarchical softmax
            if compute_loss:
                sgn = (-1.0) ** predict_word.code
                # `ch` function, 0 -> 1, 1 -> -1
                lprob = -numpy.log(expit(-sgn * prod_term))
                model.running_training_loss += sum(lprob)

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
            prod_term = numpy.dot(l1, l2b.T)
            fb = expit(prod_term)  # propagate hidden -> output
            gb = (model.neg_labels - fb) * alpha  # vector of error gradients
            # multiplied by the learning rate
            if learn_hidden:
                model.syn1neg[word_indices] += numpy.outer(gb, l1)
                # learn hidden -> output
            neu1e += numpy.dot(gb, l2b)  # save error

            # loss component corresponding to negative sampling
            if compute_loss:
                model.running_training_loss -= \
                    sum(numpy.log(expit(-1 * prod_term[1:])))
                # for the sampled words
                model.running_training_loss -= \
                    numpy.log(expit(prod_term[0]))  # for the output word

        if learn_vectors:
            if is_ft:
                model.wv.syn0_vocab[context_index[0]] += neu1e \
                    * context_locks_vocab[context_index[0]]
                for i in context_index[1:]:
                    model.wv.syn0_ngrams[i] += neu1e * context_locks_ngrams[i]
            else:
                l1 += neu1e * lock_factor  # learn input -> hidden
                # (mutates model.wv.syn0[word2.index], if that is l1)
    return neu1e


def train_batch_sg(model, sentences, alpha, work=None, compute_loss=False):
    """
    Update skip-gram model by training on a sequence of sentences.
    Each sentence is a list of string tokens, which are looked up in
    the model's vocab dictionary. Called internally from `Word2Vec.train()`.
    This is the non-optimized, Python version. If you have cython
    installed, gensim will use the optimized version from word2vec_inner
    instead.
    """
    result = 0
    window = model.window
    for sentence in sentences:
        word_vocabs = [model.wv.vocab[w] for w in sentence if w in
                       model.wv.vocab and model.wv.vocab[w].sample_int
                       > model.random.rand() * 2 ** 32 or w == '___']
        # Count the number of times that we see the nonce
        nonce_count = 0
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
                        train_sg_pair(model,
                                      model.wv.index2word[word.index],
                                      word2.index, alpha, nonce_count,
                                      compute_loss=compute_loss)

        result += len(word_vocabs)
        if window - 1 >= 3:
            window = window - model.window_decay
        model.recompute_sample_ints()
    return result


class Nonce2VecVocab(Word2VecVocab):
    def __init__(self, max_vocab_size=None, min_count=5, sample=1e-3,
                 sorted_vocab=True, null_word=0):
        super(Nonce2VecVocab, self).__init__(max_vocab_size, min_count, sample,
                                             sorted_vocab, null_word)
        self.nonce = None

    def prepare_vocab(self, hs, negative, wv, update=False,
                      keep_raw_vocab=False, trim_rule=None,
                      min_count=None, sample=None, dry_run=False):
        """Apply vocabulary settings for `min_count`.
        (discarding less-frequent words)
        and `sample` (controlling the downsampling of more-frequent words).
        Calling with `dry_run=True` will only simulate the provided
        settings and report the size of the retained vocabulary,
        effective corpus length, and estimated memory requirements.
        Results are both printed via logging and returned as a dict.
        Delete the raw vocabulary after the scaling is done to free up RAM,
        unless `keep_raw_vocab` is set.
        """
        min_count = min_count or self.min_count
        sample = sample or self.sample
        drop_total = drop_unique = 0

        if not update:
            logger.info('Loading a fresh vocabulary')
            retain_total, retain_words = 0, []
            # Discard words less-frequent than min_count
            if not dry_run:
                wv.index2word = []
                # make stored settings match these applied settings
                self.min_count = min_count
                self.sample = sample
                wv.vocab = {}

            for word, v in iteritems(self.raw_vocab):
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule):
                    retain_words.append(word)
                    retain_total += v
                    if not dry_run:
                        wv.vocab[word] = Vocab(count=v,
                                               index=len(wv.index2word))
                        wv.index2word.append(word)
                else:
                    drop_unique += 1
                    drop_total += v
            original_unique_total = len(retain_words) + drop_unique
            retain_unique_pct = len(retain_words) \
                * 100 / max(original_unique_total, 1)
            logger.info('min_count=%d retains %i unique words (%i%% of '
                        'original %i, drops %i)', min_count, len(retain_words),
                        retain_unique_pct, original_unique_total, drop_unique)
            original_total = retain_total + drop_total
            retain_pct = retain_total * 100 / max(original_total, 1)
            logger.info('min_count=%d leaves %i word corpus (%i%% of original '
                        '%i, drops %i)', min_count, retain_total, retain_pct,
                        original_total, drop_total)
        else:
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
                del wv.vocab[self.nonce]
            for word, v in iteritems(self.raw_vocab):
                # For new words, keep the ones above the min count
                # AND the nonce (regardless of count)
                if keep_vocab_item(word, v, min_count, trim_rule=trim_rule) \
                 or word == self.nonce:
                    if word in wv.vocab:
                        pre_exist_words.append(word)
                        pre_exist_total += v
                        if not dry_run:
                            wv.vocab[word].count += v
                    else:
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
            new_unique_pct = len(new_words) * 100 / max(original_unique_total,
                                                        1)
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
        elif sample < 1.0:
            # traditional meaning: set parameter as proportion of total
            threshold_count = sample * retain_total
        else:
            # new shorthand: sample >= 1 means downsample all words
            # with higher count than sample
            threshold_count = int(sample * (3 + numpy.sqrt(5)) / 2)

        downsample_total, downsample_unique = 0, 0
        for w in retain_words:
            v = self.raw_vocab[w]
            word_probability = (numpy.sqrt(v / threshold_count) + 1) \
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

        return report_values


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

    def recompute_sample_ints(self):
        for w, o in self.wv.vocab.items():
            o.sample_int = int(round(float(o.sample_int) / float(self.sample_decay)))
