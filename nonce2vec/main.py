"""Welcome to Nonce2Vec.

This is the entry point of the application.
"""

import os

import argparse
import multiprocessing
import functools
import logging
import logging.config

import math
import scipy
import numpy as np

import gensim
import spacy

from gensim.models import Word2Vec

import nonce2vec.utils.config as cutils
import nonce2vec.utils.files as futils
import nonce2vec.utils.wikipedia as wutils

from nonce2vec.models.nonce2vec import Nonce2Vec, Nonce2VecVocab, \
                                       Nonce2VecTrainables
from nonce2vec.utils.files import Samples
from nonce2vec.models.informativeness import Informativeness


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


# Note: this is scipy's spearman, without tie adjustment
def _spearman(x, y):
    return scipy.stats.spearmanr(x, y)[0]


def _get_rank(probe, nns):
    for idx, nonce_similar_word in enumerate(nns):
        word = nonce_similar_word[0]
        if word == probe:
            rank = idx + 1  # rank starts at 1
    if not rank:
        raise Exception('Could not find probe {} in nonce most similar words '
                        '{}'.format(probe, nns))
    return rank


def _update_rr_and_count(relative_ranks, count, rank):
    relative_rank = 1.0 / float(rank)
    relative_ranks += relative_rank
    count += 1
    logger.info('Rank, Relative Rank = {} {}'.format(rank, relative_rank))
    logger.info('MRR = {}'.format(relative_ranks/count))
    return relative_ranks, count


def _load_nonce2vec_model(args, info, nonce):
    logger.info('Loading Nonce2Vec model...')
    model = Nonce2Vec.load(args.background)
    model.vocabulary = Nonce2VecVocab.load(model.vocabulary)
    model.trainables = Nonce2VecTrainables.load(model.trainables)
    model.sg = 1
    #model.min_count = 1  # min_count should be the same as the background model!!
    model.replication = args.replication
    model.sum_over_set = args.sum_over_set
    model.train_over_set = args.train_over_set
    if args.sum_filter == 'random' or args.train_filter == 'random':
        model.sample = args.sample
    if args.replication:
        logger.info('Running original n2v code for replication...')
        if args.sample is None:
            raise Exception('In replication mode you need to specify the '
                            'sample parameter')
        if args.window_decay is None:
            raise Exception('In replication mode you need to specify the '
                            'window_decay parameter')
        if args.sample_decay is None:
            raise Exception('In replication mode you need to specify the '
                            'sample_decay parameter')
        model.sample_decay = args.sample_decay
        model.window_decay = args.window_decay
        model.sample = args.sample
    if not args.sum_only:
        model.train_with = args.train_with
        model.alpha = args.alpha
        model.iter = args.epochs
        model.negative = args.neg
        model.lambda_den = args.lambda_den
        model.kappa = args.kappa
        model.beta = args.beta
        model.neg_labels = []
        if model.negative > 0:
            # precompute negative labels optimization for pure-python training
            model.neg_labels = np.zeros(model.negative + 1)
            model.neg_labels[0] = 1.
    model.trainables.info = info
    model.workers = args.num_threads
    model.vocabulary.nonce = nonce
    logger.info('Model loaded')
    return model


def _test_on_chimeras(args):
    nonce = '___'
    rhos = []
    count = 0
    samples = Samples(args.dataset, source='chimeras')
    total_num_batches = sum(1 for x in samples)
    total_num_sent = sum(1 for x in [sent for batch in samples for sent in batch])
    logger.info('Testing Nonce2Vec on the chimeras dataset containing '
                '{} batches and {} sentences'.format(total_num_batches,
                                                     total_num_sent))
    num_batch = 1
    info = _load_informativeness_model(args)
    for sentences, probes, responses in samples:
        logger.info('-' * 30)
        logger.info('Processing batch {}/{}'.format(num_batch,
                                                    total_num_batches))
        num_batch += 1
        logger.info('sentences = {}'.format(sentences))
        logger.info('probes = {}'.format(probes))
        logger.info('responses = {}'.format(responses))
        model = _load_nonce2vec_model(args, info, nonce)
        vocab_size = len(model.wv.vocab)
        logger.info('vocab size = {}'.format(vocab_size))
        model.build_vocab(sentences, update=True)
        if not args.sum_only:
            model.train(sentences, total_examples=model.corpus_count,
                        epochs=model.iter)
        system_responses = []
        human_responses = []
        probe_count = 0
        for probe in probes:
            try:
                cos = model.similarity(nonce, probe)
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


def _display_stats(ranks, ctx_ents):
    logger.info('-'*30)
    logger.info('ranks stats:')
    logger.info('ranks mean = {}'.format(np.mean(ranks)))
    logger.info('ranks std = {}'.format(np.std(ranks)))
    logger.info('ranks min = {}'.format(min(ranks)))
    logger.info('ranks max = {}'.format(max(ranks)))
    logger.info('context entropy stats:')
    logger.info('ctx_ents mean = {}'.format(np.mean(ctx_ents)))
    logger.info('ctx_ents std = {}'.format(np.std(ctx_ents)))
    logger.info('ctx_ents min = {}'.format(min(ctx_ents)))
    logger.info('ctx_ents max = {}'.format(max(ctx_ents)))
    logger.info('Correlation no rounding = {}'.format(_spearman(ctx_ents,
                                                                ranks)))
    logger.info('Correlation round 6 = {}'.format(
        _spearman([round(x, 6) for x in ctx_ents], ranks)))
    logger.info('Correlation round 5 = {}'.format(
        _spearman([round(x, 5) for x in ctx_ents], ranks)))
    logger.info('Correlation round 4 = {}'.format(
        _spearman([round(x, 4) for x in ctx_ents], ranks)))
    logger.info('Correlation round 3 = {}'.format(
        _spearman([round(x, 3) for x in ctx_ents], ranks)))
    logger.info('Correlation round 2 = {}'.format(
        _spearman([round(x, 2) for x in ctx_ents], ranks)))


def _display_density_stats(ranks, sum_10, sum_25, sum_50):
    logger.info('-'*30)
    logger.info('density stats')
    logger.info('d10 rho = {}'.format(_spearman(sum_10, ranks)))
    logger.info('d25 rho = {}'.format(_spearman(sum_25, ranks)))
    logger.info('d50 rho = {}'.format(_spearman(sum_50, ranks)))


def _load_informativeness_model(args):
    if not args.info_model:
        logger.warning('Unspecified --info_model. Using background model '
                       'to compute informativeness-related probabilities')
        args.info_model = args.background
    return Informativeness(
        model_path=args.info_model, sum_filter=args.sum_filter,
        sum_thresh=args.sum_thresh, train_filter=args.train_filter,
        train_thresh=args.train_thresh, sort_by=args.sort_by)


def _compute_average_sim(sims):
    sim_sum = sum(sim[1] for sim in sims)
    return sim_sum / len(sims)


def _test_on_nonces(args):
    """Test the definitional nonces with a one-off learning procedure."""
    ranks = []
    sum_10 = []
    sum_25 = []
    sum_50 = []
    relative_ranks = 0.0
    count = 0
    samples = Samples(args.dataset, source='nonces')
    total_num_sent = sum(1 for line in samples)
    logger.info('Testing Nonce2Vec on the nonces dataset containing '
                '{} sentences'.format(total_num_sent))
    num_sent = 1
    info = _load_informativeness_model(args)
    for sentences, nonce, probe in samples:
        logger.info('-' * 30)
        logger.info('Processing sentence {}/{}'.format(num_sent,
                                                       total_num_sent))
        model = _load_nonce2vec_model(args, info, nonce)
        model.vocabulary.nonce = nonce
        vocab_size = len(model.wv.vocab)
        logger.info('vocab size = {}'.format(vocab_size))
        logger.info('nonce: {}'.format(nonce))
        logger.info('sentence: {}'.format(sentences))
        if nonce not in model.wv.vocab:
            logger.error('Nonce \'{}\' not in gensim.word2vec.model '
                         'vocabulary'.format(nonce))
            continue
        model.build_vocab(sentences, update=True)
        if not args.sum_only:
            model.train(sentences, total_examples=model.corpus_count,
                        epochs=model.iter)
        nns = model.most_similar(nonce, topn=vocab_size)
        logger.info('10 most similar words: {}'.format(nns[:10]))
        rank = _get_rank(probe, nns)
        if args.with_stats:
            ranks.append(rank)
            gold_nns = model.most_similar('{}_true'.format(nonce),
                                          topn=vocab_size)
            sum_10.append(_compute_average_sim(gold_nns[:10]))
            sum_25.append(_compute_average_sim(gold_nns[:25]))
            sum_50.append(_compute_average_sim(gold_nns[:50]))
        relative_ranks, count = _update_rr_and_count(relative_ranks, count,
                                                     rank)
        num_sent += 1
    logger.info('Final MRR =  {}'.format(relative_ranks/count))
    if args.with_stats:
        _display_density_stats(ranks, sum_10, sum_25, sum_50)


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


def _check_men(args):
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
    sentences = Samples(args.datadir, source='wiki')
    if not args.train_mode:
        raise Exception('Unspecified train mode')
    output_model_filepath = futils.get_model_path(args.datadir, args.outputdir,
                                                  args.train_mode,
                                                  args.alpha, args.neg,
                                                  args.window, args.sample,
                                                  args.epochs,
                                                  args.min_count, args.size)
    model = gensim.models.Word2Vec(
        min_count=args.min_count, alpha=args.alpha, negative=args.neg,
        window=args.window, sample=args.sample, iter=args.epochs,
        size=args.size, workers=args.num_threads)
    if args.train_mode == 'cbow':
        model.sg = 0
    if args.train_mode == 'skipgram':
        model.sg = 1
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)
    model.save(output_model_filepath)


def _test(args):
    if args.on == 'chimeras':
        _test_on_chimeras(args)
    elif args.on == 'nonces':
        _test_on_nonces(args)


def _extract(args):
    logger.info('Extracting content of wikipedia archive under {}'
                .format(args.wiki_input_dirpath))
    input_filepaths = futils.get_input_filepaths(args.wiki_input_dirpath)
    spacy_nlp = spacy.load('en_core_web_sm')
    with multiprocessing.Pool(args.num_threads) as pool:
        extract = functools.partial(wutils.extract,
                                    args.wiki_output_filepath,
                                    spacy_nlp)
        list(pool.imap_unordered(extract, input_filepaths))
    # concatenate all .txt files into single output .txt file
    logger.info('Concatenating tmp files...')
    tmp_filepaths = futils.get_tmp_filepaths(args.wiki_output_filepath)
    with open(args.wiki_output_filepath, 'w', encoding='utf-8') as output_stream:
        for tmp_filepath in tmp_filepaths:
            with open(tmp_filepath, 'r') as tmp_stream:
                for line in tmp_stream:
                    print(line, file=output_stream)
    logger.info('Done extracting content of Wikipedia archive')
    shutil.rmtree(futils.get_tmp_dirpath(args.wiki_output_filepath))


def main():
    """Launch Nonce2Vec."""
    parser = argparse.ArgumentParser(prog='nonce2vec')
    subparsers = parser.add_subparsers()
    # a shared set of parameters when using gensim
    parser_gensim = argparse.ArgumentParser(add_help=False)
    parser_gensim.add_argument('--num_threads', type=int, default=1,
                               help='number of threads to be used by gensim')
    parser_gensim.add_argument('--alpha', type=float,
                               help='initial learning rate')
    parser_gensim.add_argument('--neg', type=int,
                               help='number of negative samples')
    parser_gensim.add_argument('--window', type=int,
                               help='window size')
    parser_gensim.add_argument('--sample', type=float,
                               help='subsampling rate')
    parser_gensim.add_argument('--epochs', type=int,
                               help='number of epochs')
    parser_gensim.add_argument('--min_count', type=int,
                               help='min frequency count')

    # a shared set of parameters when using informativeness
    parser_info = argparse.ArgumentParser(add_help=False)
    parser_info.add_argument('--info_model', type=str,
                             help='informativeness model path')
    parser_info.add_argument('--sum_filter', default=None,
                             choices=['random', 'self', 'cwi'],
                             help='filter for sum initialization')
    parser_info.add_argument('--sum_threshold', type=int,
                             dest='sum_thresh',
                             help='sum filter threshold for self and cwi')
    parser_info.add_argument('--train_filter', default=None,
                             choices=['random', 'self', 'cwi'],
                             help='filter over training context')
    parser_info.add_argument('--train_threshold', type=int,
                             dest='train_thresh',
                             help='train filter threshold for self and cwi')
    parser_info.add_argument('--sort_by', choices=['asc', 'desc'],
                             default=None,
                             help='cwi sorting order for context items')

    # train word2vec with gensim from a wikipedia dump
    parser_train = subparsers.add_parser(
        'train', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_gensim],
        help='generate pre-trained embeddings from wikipedia dump via '
             'gensim.word2vec')
    parser_train.set_defaults(func=_train)
    parser_train.add_argument('--data', required=True, dest='datadir',
                              help='absolute path to training data directory')
    parser_train.add_argument('--size', type=int, default=400,
                              help='vector dimensionality')
    parser_train.add_argument('--train_mode', choices=['cbow', 'skipgram'],
                              help='how to train word2vec')
    parser_train.add_argument('--outputdir', required=True,
                              help='Absolute path to outputdir to save model')

    # check various metrics
    parser_check = subparsers.add_parser(
        'check', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_info],
        help='check w2v embeddings quality by calculating correlation with '
             'the similarity ratings in the MEN dataset. Also, check the '
             'distribution of context_entropy across datasets')
    parser_check.set_defaults(func=_check_men)
    parser_check.add_argument('--data', required=True, dest='men_dataset',
                              help='absolute path to dataset')
    parser_check.add_argument('--model', required=True, dest='w2v_model',
                              help='absolute path to the word2vec model')

    # test nonce2vec in various config on the chimeras and nonces datasets
    parser_test = subparsers.add_parser(
        'test', formatter_class=argparse.RawTextHelpFormatter,
        parents=[parser_gensim, parser_info],
        help='test nonce2vec')
    parser_test.set_defaults(func=_test)
    parser_test.add_argument('--on', required=True,
                             choices=['nonces', 'chimeras'],
                             help='type of test data to be used')
    parser_test.add_argument('--model', required=True,
                             dest='background',
                             help='absolute path to word2vec pretrained model')
    parser_test.add_argument('--data', required=True, dest='dataset',
                             help='absolute path to test dataset')
    parser_test.add_argument('--train_with',
                             choices=['exp_alpha', 'cwi_alpha', 'cst_alpha'],
                             help='learning rate computation function')
    parser_test.add_argument('--lambda', type=float,
                             dest='lambda_den',
                             help='lambda decay')
    parser_test.add_argument('--kappa', type=int,
                             help='kappa')
    parser_test.add_argument('--beta', type=int,
                             help='beta')
    parser_test.add_argument('--sample_decay', type=float,
                             help='sample decay')
    parser_test.add_argument('--window_decay', type=int,
                             help='window decay')
    parser_test.add_argument('--sum_only', action='store_true', default=False,
                             help='sum only: no additional training after sum initialization')
    parser_test.add_argument('--replication', action='store_true', default=False,
                             help='use original n2v code')
    parser_test.add_argument('--sum_over_set', action='store_true', default=False,
                             help='sum over set of context items rather than list')
    parser_test.add_argument('--train_over_set', action='store_true', default=False,
                             help='train over set of context items rather than list')
    parser_test.add_argument('--with_stats', action='store_true', default=False,
                             help='display informativeness statistics alongside test results')

    # extract data from Wikipedia XML dump and convert to UTF-8
    # lowercase 1 sentence-per-line format.
    parser_extract = subparsers.add_parser(
        'extract', formatter_class=argparse.RawTextHelpFormatter,
        help='extract content from Wikipedia XML dump')
    parser_extract.set_defaults(func=_extract)
    parser_extract.add_argument('-i', '--input', required=True,
                                dest='wiki_input_dirpath',
                                help='absolute path to directory containing Wikipedia XML files')
    parser_extract.add_argument('-o', '--output', required=True,
                                dest='wiki_output_filepath',
                                help='absolute path to output .txt file')
    parser_extract.add_argument('-n', '--num_threads', type=int,
                                required=False, default=1,
                                dest='num_threads',
                                help='number of CPU threads to be used')
    args = parser.parse_args()
    args.func(args)
