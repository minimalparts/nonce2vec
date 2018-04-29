"""Welcome to Nonce2Vec.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

import numpy

import gensim
from gensim.models import Word2Vec

from nonce2vec.models.nonce2vec import Nonce2Vec, Nonce2VecVocab, Nonce2VecTrainables

import nonce2vec.utils.config as cutils
import nonce2vec.utils.files as futils
from nonce2vec.utils.files import Sentences


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _update_mrr_and_count(mrr, count, nns, probe):
    rr = 0
    n = 1
    for nn in nns:
        word = nn[0]
        if word == probe:
            logger.info('probe: {}'.format(word))
            rr = n
        else:
            n += 1
    if rr != 0:
        mrr += 1.0 / float(rr)
    count += 1
    logger.info('RR, MRR = {} {}'.format(rr, mrr))
    logger.info('MRR = {}'.format(mrr/count))
    return mrr, count


def _load_nonce2vec_model(background, alpha, sample, neg, window, iteration,
                          lambda_den, sample_decay, window_decay, num_threads):
    logger.info('Loading Nonce2Vec model...')
    model = Nonce2Vec.load(background)
    w2vec_vocab = model.vocabulary
    n2vec_vocab = Nonce2VecVocab()
    w2vec_trainables = model.trainables
    n2vec_trainables = Nonce2VecTrainables()
    for key, value in w2vec_vocab.__dict__.items():
        setattr(n2vec_vocab, key, value)
    for key, value in w2vec_trainables.__dict__.items():
        setattr(n2vec_trainables, key, value)
    model.vocabulary = n2vec_vocab
    model.trainables = n2vec_trainables
    model.sg = 1
    model.alpha = alpha
    model.sample = sample
    model.sample_decay = sample_decay
    model.iter = iteration
    model.negative = neg
    model.window = window
    model.window_decay = window_decay
    model.lambda_den = lambda_den
    model.min_count = 1
    model.workers = num_threads
    model.neg_labels = []
    if model.negative > 0:
        # precompute negative labels optimization for pure-python training
        model.neg_labels = numpy.zeros(model.negative + 1)
        model.neg_labels[0] = 1.
    logger.info('Model loaded')
    return model


def _test_chimeras():
    pass


def _test_def_nonces(args):
    """Test the definitional nonces with a one-off learning procedure."""
    mrr = 0.0
    count = 0
    with open(args.dataset, 'r') as datastream:
        total_num_sent = sum(1 for line in datastream)
        logger.info('Testing Nonce2Vec on the definitional dataset containing '
                    '{} sentences'.format(total_num_sent))
        num_sent = 1
        datastream.seek(0)
        for line in datastream:
            logger.info('-' * 30)
            logger.info('Processing sentence {}/{}'.format(num_sent,
                                                           total_num_sent))
            model = _load_nonce2vec_model(args.background, args.alpha,
                                          args.sample, args.neg, args.window,
                                          args.iteration, args.lambda_den,
                                          args.sample_decay, args.window_decay,
                                          args.num_threads)
            vocab_size = len(model.wv.vocab)
            logger.error('vocab size = {}'.format(vocab_size))
            fields = line.rstrip('\n').split('\t')
            nonce = fields[0]
            sentence = fields[1].replace('___', nonce).split()
            probe = '{}_true'.format(nonce)
            logger.info('nonce: {}'.format(nonce))
            logger.info('sentence: {}'.format(sentence))
            if nonce not in model.wv.vocab:
                logger.error('Nonce \'{}\' not in gensim.word2vec.model '
                             'vocabulary'.format(nonce))
                continue
            model.vocabulary.nonce = nonce
            model.build_vocab([sentence], update=True)
            model.train([sentence], total_examples=model.corpus_count,
                        epochs=model.iter)

            nns = model.most_similar(nonce, topn=vocab_size)
            logger.info('10 most similar words: {}'.format(nns[:10]))
            mrr, count = _update_mrr_and_count(mrr, count, nns, probe)
            num_sent += 1
        logger.info('Final MRR =  {}'.format(mrr/count))


def _test(args):

    if args.mode == 'def_nonces':
        _test_def_nonces(args)
    if args.mode == 'chimeras':
        _test_chimeras()


def _train(args):
    sentences = Sentences(args.datadir)
    output_model_filepath = futils.get_model_path(args.datadir, args.outputdir)
    model = gensim.models.Word2Vec(
        min_count=args.min_count, alpha=args.alpha, negative=args.negative,
        window=args.window, sample=args.sample, iter=args.epochs,
        size=args.size, workers=args.num_threads)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save(output_model_filepath)


def main():
    """Launch Nonce2Vec."""
    parser = argparse.ArgumentParser(prog='nonce2vec')
    subparsers = parser.add_subparsers()
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        help='generate pre-trained embeddings from wikipedia dump via '
             'gensim.word2vec')
    parser_train.set_defaults(func=_train)
    parser_train.add_argument('--data', required=True,
                              dest='datadir',
                              help='absolute path to training data directory')
    parser_train.add_argument('--num_threads',
                              type=int, default=1,
                              help='number of threads to be used by gensim')
    parser_train.add_argument('--alpha',
                              type=float, default=0.025,
                              help='initial learning rate')
    parser_train.add_argument('--negative',
                              type=int, default=5,
                              help='number of negative samples')
    parser_train.add_argument('--window',
                              type=int, default=5,
                              help='window size')
    parser_train.add_argument('--sample',
                              type=float, default=1e-3,
                              help='subsampling')
    parser_train.add_argument('--epochs',
                              type=int, default=5,
                              help='number of epochs')
    parser_train.add_argument('--min_count',
                              type=int, default=50,
                              help='min frequency count')
    parser_train.add_argument('--size',
                              type=int, default=400,
                              help='vector dimensionality')
    parser_train.add_argument('--outputdir',
                              required=True,
                              help='Absolute path to outputdir to save model')
    parser_test = subparsers.add_parser(
        'test', formatter_class=argparse.RawTextHelpFormatter,
        help='test nonce2vec')
    parser_test.set_defaults(func=_test)
    parser_test.add_argument('--mode', required=True,
                             choices=['def_nonces', 'chimeras'],
                             help='what is to be tested')
    parser_test.add_argument('--model', required=True,
                             dest='background',
                             help='gensim model generated from wikipedia')
    parser_test.add_argument('--dataset', required=True,
                             help='')
    parser_test.add_argument('--alpha', required=True, type=float,
                             help='')
    parser_test.add_argument('--sample', required=True, type=float,
                             help='')
    parser_test.add_argument('--neg', required=True, type=int,
                             help='')
    parser_test.add_argument('--window', required=True, type=int,
                             help='')
    parser_test.add_argument('--iter', required=True, type=int,
                             dest='iteration',
                             help='')
    parser_test.add_argument('--lambda', required=True, type=float,
                             dest='lambda_den',
                             help='')
    parser_test.add_argument('--sample_decay', required=True, type=float,
                             help='')
    parser_test.add_argument('--window_decay', required=True, type=int,
                             help='')
    parser_test.add_argument('--num_threads',
                             type=int, default=1,
                             help='number of threads to be used by gensim')
    args = parser.parse_args()
    args.func(args)
