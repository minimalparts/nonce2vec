import numpy

from nonce2vec.models.informativeness import Informativeness
from nonce2vec.utils.files import Sentences


def _test_context_entropy():
    model_path = '/Users/AKB/Github/nonce2vec/models/wiki_all.sent.split.model'
    info = Informativeness(mode='cbow', model_path=model_path, entropy='shannon')
    sent1 = 'gynecology is a branch of medicine '

def _test_nonces():
    model_path = '/Users/AKB/Github/nonce2vec/models/wiki_all.sent.split.model'
    data_path = '/Users/AKB/Github/nonce2vec/data/definitions/nonce.definitions.300.test'
    info = Informativeness(mode='cbow', model_path=model_path)
    sentences = Sentences(data_path, source='nonce_or_chimera')
    for fields in sentences:
        nonce = fields[0]
        tokens = fields[1].replace('___', nonce).split()
        #tokens = fields[1].split()
        #print(tokens)
        #print(info.sentence2word(tokens, 0))
        for idx, token in enumerate(tokens):
            if idx > 0:
                print('{} | {}'.format(token, info.word2word(tokens, idx, 0)))
        #_tokens = [token for token in tokens if numpy.log(info._model.wv.vocab[token].sample_int) > 22]
        print(' ')
        # for idx, token in enumerate(_tokens):
        #     if idx > 0:
        #         print('{} | {}'.format(token, info.word2word(tokens, idx, 0)))


def _test_chimeras():
    model_path = '/Users/AKB/Github/nonce2vec/models/wiki_all.sent.split.model'
    data_path = '/Users/AKB/Github/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt'
    info = Informativeness(mode='cbow', model_path=model_path, entropy='shannon')
    # sent_1 = 'The cat was chasing the mouse around the house'
    # sent_2 = 'The cat was chasing the rabbit around the house'
    # #sent_2 = 'The cat argued about the politics of the united states, freedom and the bill of rights'
    # print(info.sentence2sentence(sent_1.split(' '), 1, sent_2.split(), 1))
    _sentences = Sentences(data_path, source='nonce_or_chimera')
    for fields in _sentences:
        sentences = []
        for sent in fields[1].split('@@'):
            sentences.append(sent.strip().split(' '))
            tokens = sent.strip().split(' ')
            print(tokens)
            if '___' in tokens:
                print('s2w = {}'.format(info.sentence2word(tokens, tokens.index('___'))))
        # print('sent_1: {}'.format(sentences[0]))
        # print('sent_2: {}'.format(sentences[1]))
        # print('sent_3: {}'.format(sentences[2]))
        # print('s2s12 = {}'.format(info.sentence2sentence(sentences[0], sentences[0].index('___'),
        #                                                  sentences[1], sentences[1].index('___'))))
        # print('s2s13 = {}'.format(info.sentence2sentence(sentences[0], sentences[0].index('___'),
        #                                                  sentences[2], sentences[2].index('___'))))
        # print('s2s23 = {}'.format(info.sentence2sentence(sentences[1], sentences[1].index('___'),
        #                                                  sentences[2], sentences[2].index('___'))))

if __name__ == '__main__':
    #_test_nonces()
    #_test_chimeras()
    _test_context_entropy()
