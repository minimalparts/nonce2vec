"""Welcome to Nonce2Vec.

This is the entry point of the application.
"""

import os

import argparse
import logging
import logging.config

import gensim
from gensim.models import Word2Vec

import nonce2vec.utils.config as cutils
import nonce2vec.utils.files as futils
from nonce2vec.utils.files import Sentences


logging.config.dictConfig(
    cutils.load(
        os.path.join(os.path.dirname(__file__), 'logging', 'logging.yml')))

logger = logging.getLogger(__name__)


def _update_mrr_and_count(mrr, count, nns, probe):
    rr = 0.0
    n = 1.0
    for nn in nns:
        word = nn[0]
        if word == probe:
            print('probe: {}'.format(word))
            rr = n
        else:
            n += 1.0
    if rr != 0.0:
        mrr += 1.0 / rr
    count += 1.0
    print('RR, MRR = {} {}'.format(rr, mrr))
    print('MRR = {}'.format(mrr/count))
    return mrr, count


def _load_nonce2vec_model(background, alpha, sample, neg, window, iteration,
                          lambda_den, sample_decay, window_decay):
    print('Loading Word2Vec model...')
    model = Word2Vec.load(background)
    model.alpha = alpha
    model.sample = sample
    model.sample_decay = sample_decay
    model.iter = iteration
    model.negative = neg
    model.window = window
    model.window_decay = window_decay
    model.lambda_den = lambda_den
    model.min_count = 1
    print('Model loaded')
    return model


def _test_chimeras():
    pass


def _test_def_nonces(dataset, model):
    """Test the definitional nonces with a one-off learning procedure."""
    mrr = 0.0
    count = 0
    vocab_size = len(model.wv.vocab)
    with open(dataset, 'r') as datastream:
        total_num_sent = sum(1 for line in datastream)
        print('Testing Nonce2Vec on the definitional dataset containing {} '
              'sentences'.format(total_num_sent))
        num_sent = 1
        datastream.seek(0)
        for line in datastream:
            print('-' * 80)
            print('Processing sentence {}/{}'.format(num_sent, total_num_sent))
            fields = line.rstrip('\n').split('\t')
            nonce = fields[0]
            sentence = fields[1].replace('___', nonce).split()
            probe = '{}_true'.format(nonce)
            print('nonce: {}'.format(nonce))
            print('sentence: {}'.format(sentence))
            if nonce not in model.wv.vocab:
                print('Nonce \'{}\' not in gensim.word2vec.model vocabulary'
                      .format(nonce))
                continue
            model.nonce = nonce
            model.build_vocab([sentence], update=True)
            model.train(sentence)
            nns = model.most_similar(nonce, topn=vocab_size)
            print('10 most similar words: {}'.format(nns[:10]))
            mrr, count = _update_mrr_and_count(mrr, count, nns, probe)
            num_sent += 1
        print('Final MRR =  {}'.format(mrr/count))

    # mrr = 0.0
    #
    # human_responses = []
    # system_responses = []
    #
    # c = 0
    # ranks = []
    # f=open(dataset)
    # for l in f:
    #     fields=l.rstrip('\n').split('\t')
    #     nonce = fields[0]
    #     sentence = [fields[1].replace("___",nonce).split()]
    #     probe = nonce+"_true"
    #     #model = Word2Vec.load(background)
    #     vocab_size = len(model.wv.vocab)
    #     if nonce in model.wv.vocab:
    #         model.nonce = nonce
    #         model.build_vocab(sentence, update=True)
    #         model.min_count=1
    #         model.train(sentence)
    #         nns = model.most_similar(nonce,topn=vocab_size)
    #
    #         rr = 0
    #         n = 1
    #         for nn in nns:
    #             w = nn[0]
    #             if w == probe:
    #                 rr = n
    #                 ranks.append(rr)
    #             else:
    #               n+=1
    #
    #         if rr != 0:
    #             mrr+=float(1)/float(rr)
    #         print(rr,mrr)
    #         c+=1
    #     else:
    #       print("nonce not known...")


def _test(args):
    model = _load_nonce2vec_model(args.background, args.alpha,
                                  args.sample, args.neg, args.window,
                                  args.iteration, args.lambda_den,
                                  args.sample_decay, args.window_decay)
    if args.mode == 'def_nonces':
        _test_def_nonces(args.dataset, model)
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
    args = parser.parse_args()
    args.func(args)
