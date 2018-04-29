"""This test the definitional nonces with a one-off learning procedure.

Crazy learning rate.

python test_def_nonces.py models/wiki_all.sent.split.model
                          data/definitions/nonce.definitions.300.test
                          1 10000 3 15 1 70 1.9 5
"""

import numpy as np
import collections
import sys
from gensim.models import Word2Vec

background = sys.argv[1]
dataset = sys.argv[2]

alpha = sys.argv[3]
sample = sys.argv[4]
neg = sys.argv[5]
window = sys.argv[6]
iter = sys.argv[7]
lambda_den = sys.argv[8]
sample_decay = sys.argv[9]
window_decay = sys.argv[10]
mrr = 0.0

human_responses = []
system_responses = []

c = 0
ranks = []
f=open(dataset)
for l in f:
    fields=l.rstrip('\n').split('\t')
    nonce = fields[0]
    sentence = [fields[1].replace("___",nonce).split()]
    probe = nonce+"_true"
    print "--"
    print nonce
    print sentence
    model = Word2Vec.load(background)
    vocab_size = len(model.wv.vocab)
    print('vocab_size = ', vocab_size)
    #model.random = np.random.RandomState(seed=1) # FIXME: hack
    #print(model.random.rand())
    if nonce in model.wv.vocab:
        model.alpha=float(alpha)
        model.sample=float(sample)
        model.sample_decay=float(sample_decay)
        model.iter=int(iter)
        model.negative=int(neg)
        model.nonce=nonce
        model.window=int(window)
        model.window_decay=int(window_decay)
        model.lambda_den=float(lambda_den)
        model.build_vocab(sentence, update=True)
        model.min_count=1
        model.train(sentence)
        nns = model.most_similar(nonce,topn=vocab_size)

        print nns[:10]
        rr = 0
        n = 1
        for nn in nns:
            w = nn[0]
            if w == probe:
                print w
                rr = n
                ranks.append(rr)
            else:
              n+=1

        if rr != 0:
            mrr+=float(1)/float(rr)
        print rr,mrr
        c+=1
    else:
      print "nonce not known..."

f.close()

print "Final MRR: ",mrr,c,float(mrr)/float(c)

bins = np.linspace(0,vocab_size,40)
print bins
binned = np.digitize(ranks, bins)
print collections.Counter(binned)
