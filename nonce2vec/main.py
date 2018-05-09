"""Welcome to Nonce2Vec.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

import math
import scipy
import numpy as np

import gensim

import nonce2vec.utils.config as cutils
import nonce2vec.utils.files as futils

from gensim.models import Word2Vec
from nonce2vec.models.nonce2vec import Nonce2Vec, Nonce2VecVocab, \
                                       Nonce2VecTrainables
from nonce2vec.utils.files import Sentences
from nonce2vec.models.informativeness import Informativeness


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


# Note: this is scipy's spearman, without tie adjustment
def _spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def _update_rr_and_count(relative_ranks, count, nns, probe):
    for idx, nonce_similar_word in enumerate(nns):
        word = nonce_similar_word[0]
        if word == probe:
            logger.info('probe: {}'.format(word))
            rank = idx + 1  # rank starts at 1
    if not rank:
        raise Exception('Could not find probe {} in nonce most similar words '
                        '{}'.format(probe, nns))
    relative_rank = 1.0 / float(rank)
    relative_ranks += relative_rank
    count += 1
    logger.info('Rank, Relative Rank = {} {}'.format(rank, relative_rank))
    logger.info('MRR = {}'.format(relative_ranks/count))
    return relative_ranks, count



def _load_nonce2vec_model(background, alpha, sample, neg, window, epochs,
                          lambda_den, sample_decay, window_decay,
                          num_threads):
    logger.info('Loading Nonce2Vec model...')
    model = Nonce2Vec.load(background)
    model.vocabulary = Nonce2VecVocab.load(model.vocabulary)
    model.trainables = Nonce2VecTrainables.load(model.trainables)
    model.sg = 1
    model.alpha = alpha
    model.sample = sample
    model.sample_decay = sample_decay
    model.iter = epochs
    model.negative = neg
    model.window = window
    model.window_decay = window_decay
    model.lambda_den = lambda_den
    model.workers = num_threads
    model.neg_labels = []
    if model.negative > 0:
        # precompute negative labels optimization for pure-python training
        model.neg_labels = np.zeros(model.negative + 1)
        model.neg_labels[0] = 1.
    logger.info('Model loaded')
    return model


def _test_chimeras(args):
    _sentences = Sentences(args.dataset, source='nonce_or_chimera')
    nonce = '___'
    rhos = []
    count = 0
    for fields in _sentences:
        sentences = []
        for sent in fields[1].split('@@'):
            sentences.append(sent.split(' '))
        probes = fields[2].split(',')
        responses = fields[3].split(',')
        logger.info('-' * 30)
        logger.info('sentences = {}'.format(sentences))
        logger.info('probes = {}'.format(probes))
        logger.info('responses = {}'.format(responses))
        model = _load_nonce2vec_model(args.background, args.alpha,
                                      args.sample, args.neg, args.window,
                                      args.epochs,
                                      args.lambda_den,
                                      args.sample_decay, args.window_decay,
                                      args.num_threads)
        vocab_size = len(model.wv.vocab)
        logger.info('vocab size = {}'.format(vocab_size))
        model.vocabulary.nonce = nonce
        model.build_vocab(sentences, update=True)
        model.min_count = args.min_count
        if not args.sum_only:
            model.train(sentences, total_examples=model.corpus_count,
                        epochs=model.iter)
        system_responses = []
        human_responses = []
        probe_count = 0
        for probe in probes:
            try:
                cos = model.similarity('___', probe)
                system_responses.append(cos)
                human_responses.append(responses[probe_count])
            except:
                logger.error('ERROR processing probe {}'.format(probe))
            probe_count += 1
        if len(system_responses) > 1:
            logger.info('system_responses = {}'.format(system_responses))
            logger.info('human_responses = {}'.format(human_responses))
            logger.info('10 most similar words = {}'.format(
                model.most_similar(nonce, topn=10)))
            rho = _spearman(human_responses, system_responses)
            logger.info('RHO = {}'.format(rho))
            if not math.isnan(rho):
                rhos.append(rho)
        count += 1
    logger.info('AVERAGE RHO = {}'.format(float(sum(rhos))/float(len(rhos))))


def _test_nonces(args):
    """Test the definitional nonces with a one-off learning procedure."""
    relative_ranks = 0.0
    count = 0
    sentences = Sentences(args.dataset, source='nonce_or_chimera')
    total_num_sent = sum(1 for line in sentences)
    logger.info('Testing Nonce2Vec on the definitional dataset containing '
                '{} sentences'.format(total_num_sent))
    num_sent = 1
    if args.with_info:
        informativeness = Informativeness(
            mode=args.info_mode, w2v_model_path=args.info_model_path,
            entropy=args.entropy)
    for fields in sentences:
        logger.info('-' * 30)
        logger.info('Processing sentence {}/{}'.format(num_sent,
                                                       total_num_sent))
        model = _load_nonce2vec_model(args.background, args.alpha,
                                      args.sample, args.neg, args.window,
                                      args.epochs,
                                      args.lambda_den,
                                      args.sample_decay, args.window_decay,
                                      args.num_threads)
        vocab_size = len(model.wv.vocab)
        logger.info('vocab size = {}'.format(vocab_size))
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
        if args.with_info:
            model.build_vocab([sentence], filters=args.filters,
                              self_info_threshold=args.self_thresh, update=True,
                              informativeness=informativeness)
        else:
            model.build_vocab([sentence], filters=args.filters,
                              self_info_threshold=args.self_thresh, update=True)
        model.min_count = args.min_count
        if not args.sum_only:
            model.train([sentence], total_examples=model.corpus_count,
                        epochs=model.iter)
        nns = model.most_similar(nonce, topn=vocab_size)
        logger.info('10 most similar words: {}'.format(nns[:10]))
        relative_ranks, count = _update_rr_and_count(relative_ranks, count,
                                                     nns, probe)
        num_sent += 1
    logger.info('Final MRR =  {}'.format(relative_ranks/count))


def _get_men_pairs_and_sim(men_dataset):
    pairs = []
    humans = []
    with open(men_dataset, 'r') as men_stream:
        for line in men_stream:
            line = line.rstrip('\n')
            items = line.split()
            pairs.append((items[0], items[1]))
            humans.append(float(items[2]))
    return pairs, humans


def _cosine_similarity(peer_v, query_v):
    if len(peer_v) != len(query_v):
        raise ValueError('Vectors must be of same length')
    num = np.dot(peer_v, query_v)
    den_a = np.dot(peer_v, peer_v)
    den_b = np.dot(query_v, query_v)
    return num / (math.sqrt(den_a) * math.sqrt(den_b))


def _test_men(args):
    """Check embeddings quality.

    Calculate correlation with the similarity ratings in the MEN dataset.
    """
    logger.info('Checking embeddings quality against MEN similarity ratings')
    pairs, humans = _get_men_pairs_and_sim(args.men_dataset)
    logger.info('Loading word2vec model...')
    model = Word2Vec.load(args.w2v_model)
    logger.info('Model loaded')
    system_actual = []
    human_actual = []  # This is needed because we may not be able to
                       # calculate cosine for all pairs
    count = 0
    for (first, second), human in zip(pairs, humans):
        if first not in model.wv.vocab or second not in model.wv.vocab:
            logger.error('Could not find one of more pair item in model '
                         'vocabulary: {}, {}'.format(first, second))
            continue
        sim = _cosine_similarity(model.wv[first], model.wv[second])
        system_actual.append(sim)
        human_actual.append(human)
        count += 1
    spr = _spearman(human_actual, system_actual)
    logger.info('SPEARMAN: {} calculated over {} items'.format(spr, count))


def _train(args):
    sentences = Sentences(args.datadir, source='wiki')
    output_model_filepath = futils.get_model_path(args.datadir, args.outputdir)
    model = gensim.models.Word2Vec(
        min_count=args.min_count, alpha=args.alpha, negative=args.neg,
        window=args.window, sample=args.sample, iter=args.epochs,
        size=args.size, workers=args.num_threads)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save(output_model_filepath)


def _test(args):
    if args.mode == 'nonces':
        _test_nonces(args)
    if args.mode == 'chimeras':
        _test_chimeras(args)


def main():
    """Launch Nonce2Vec."""
    parser = argparse.ArgumentParser(prog='nonce2vec')
    subparsers = parser.add_subparsers()
    # a shared set of parameters
    parser_template = argparse.ArgumentParser(add_help=False)
    parser_template.add_argument('--num_threads',
                                 type=int, default=1,
                                 help='number of threads to be used by gensim')
    parser_template.add_argument('--alpha', type=float, default=0.025,
                                 help='initial learning rate')
    parser_template.add_argument('--neg', required=True, type=int,
                                 help='number of negative samples')
    parser_template.add_argument('--window', type=int, default=5,
                                 help='window size')
    parser_template.add_argument('--sample', type=float, default=1e-3,
                                 help='subsampling')
    parser_template.add_argument('--epochs', type=int, default=5,
                                 help='number of epochs')
    parser_template.add_argument('--min_count', type=int, default=50,
                                 help='min frequency count')
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_template],
        help='generate pre-trained embeddings from wikipedia dump via '
             'gensim.word2vec')
    parser_train.set_defaults(func=_train)
    parser_train.add_argument('--data', required=True, dest='datadir',
                              help='absolute path to training data directory')
    parser_train.add_argument('--size',
                              type=int, default=400,
                              help='vector dimensionality')
    parser_train.add_argument('--outputdir',
                              required=True,
                              help='Absolute path to outputdir to save model')
    parser_test_men = subparsers.add_parser(
        'men', formatter_class=argparse.RawTextHelpFormatter,
        help='check w2v embeddings quality by calculating correlation with '
             'the similarity ratings in the MEN dataset')
    parser_test_men.set_defaults(func=_test_men)
    parser_test_men.add_argument('--data', required=True, dest='men_dataset',
                                 help='absolute path to MEN dataset')
    parser_test_men.add_argument('--model', required=True, dest='w2v_model',
                                 help='absolute path to the word2vec model')
    parser_test = subparsers.add_parser(
        'test', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_template],
        help='test nonce2vec')
    parser_test.set_defaults(func=_test)
    parser_test.add_argument('--mode', required=True,
                             choices=['nonces', 'chimeras'],
                             help='what is to be tested')
    parser_test.add_argument('--model', required=True,
                             dest='background',
                             help='absolute path to word2vec pretrained model')
    parser_test.add_argument('--data', required=True, dest='dataset',
                             help='')
    parser_test.add_argument('--lambda', required=True, type=float,
                             dest='lambda_den',
                             help='')
    parser_test.add_argument('--sample_decay', required=True, type=float,
                             help='')
    parser_test.add_argument('--window_decay', required=True, type=int,
                             help='')
    parser_test.add_argument('--filters',
                             choices=['random', 'self', 'w2w'],
                             help='')
    parser_test.add_argument('--self_info_threshold', type=int,
                             dest='self_thresh',
                             help='')
    parser_test.add_argument('--info_mode', choices=['cbow', 'bidir'],
                             help='Informativeness mode. How probabilities are gathered')
    parser_test.add_argument('--info_model', type=str,
                             dest='info_model_path',
                             help='Informativeness model path')
    parser_test.add_argument('--info_entropy', choices=['shannon', 'weighted'],
                             dest='entropy',
                             help='entropy')
    parser_test.add_argument('--sum_only', action='store_true', default=False,
                             help='')
    parser_test.add_argument('--with_info', action='store_true', default=False,
                             help='')
    args = parser.parse_args()
    args.func(args)
