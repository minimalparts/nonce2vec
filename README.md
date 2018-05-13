[![DOI](https://zenodo.org/badge/96074751.svg)](https://zenodo.org/badge/latestdoi/96074751)

# nonce2vec
This is the repo accompanying the paper "High-risk learning: acquiring new word
vectors from tiny data" (Herbelot &amp; Baroni, 2017). If you use this code,
please cite the following:
```tex
@InProceedings{herbelot-baroni:2017:EMNLP2017,
  author    = {Herbelot, Aur\'{e}lie  and  Baroni, Marco},
  title     = {High-risk learning: acquiring new word vectors from tiny data},
  booktitle = {Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  month     = {September},
  year      = {2017},
  address   = {Copenhagen, Denmark},
  publisher = {Association for Computational Linguistics},
  pages     = {304--309},
  url       = {https://www.aclweb.org/anthology/D17-1030}
}
```

## Replication HowTo
The following sections detail the steps required to replicate results from scratch.

### Install nonce2vec
Under the nonce2vec directory, run:
```bash
sudo -H python3 setup.py develop
```

### Download and extract required resources


### Generate a pre-trained word2vec model
To generate a gensim.word2vec model, run:
```bash
n2v train \
  --data /absolute/path/to/wikipedia/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min_count 50 \
  --size 400 \
  --num_threads number_of_threads_available_in_your_env
```

### Check the correlation with the MEN dataset
To check the quality of your pre-trained gensim.word2vec model
against the MEN dataset, run:
```bash
n2v men \
  --data /absolute/path/to/MEN/MEN_dataset_natural_form_full
  --model /absolute/path/to/gensim/word2vec/model
```

### Test nonce2vec on the nonce definitional dataset
```bash
n2v test \
  --mode nonces \
  --model /absolute/path/to/pretrained/w2v/model \
  --data /absolute/path/to/nonce.definitions.300.test \
  --alpha 1 \
  --neg 3 \
  --window 15 \
  --sample 10000 \
  --epochs 1 \
  --min_count 1 \
  --lambda 70 \
  --sample_decay 1.9 \
  --window_decay 5 \
  --num_threads number_of_threads_available_in_your_env
```
To test in *sum_only* mode which just sums overs pre-existing vectors, just add
the `sum_only` flag
```bash
n2v test \
  --mode nonces \
  --model /absolute/path/to/pretrained/w2v/model \
  --data /absolute/path/to/nonce.definitions.300.test \
  --alpha 1 \
  --neg 3 \
  --window 15 \
  --sample 10000 \
  --epochs 1 \
  --min_count 1 \
  --lambda 70 \
  --sample_decay 1.9 \
  --window_decay 5 \
  --num_threads number_of_threads_available_in_your_env \
  --sum_only
```


### Test nonce2vec on the chimera dataset
```bash
n2v test \
  --mode chimeras \
  --model /absolute/path/to/pretrained/w2v/model \
  --data /absolute/path/to/chimeras.dataset.lx.tokenised.test.txt \
  --alpha 1 \
  --neg 3 \
  --window 15 \
  --sample 10000 \
  --epochs 1 \
  --min_count 1 \
  --lambda 70 \
  --sample_decay 1.9 \
  --window_decay 5 \
  --num_threads number_of_threads_available_in_your_env
```

## Replication results
Details regarding the pre-pretrained w2v models:

| pre-trained model | vocab size | MEN | Details |
| --- | --- | --- | --- |
| `wiki_all.sent.split.model` | 259376 | 0.7496 | Aurélie's wikidump |
| `wikidump.w2v.model` | 274449 | 0.7032 | Alex's wikidump (lowercase UTF-8 version of Aurélie's) |

On the nonce dataset:

| pre-trained model | MRR |
| --- | --- |
| `wiki_all.sent.split.model` | 0.049172107209112415 |
| `wikidump.w2v.model` | 0.03244086200454485 |

in *sum_only* mode:

| pre-trained model | MRR |
| --- | --- |
| `wiki_all.sent.split.model` | 0.041074614290738116 |
| `wikidump.w2v.model` | 0.03029920582998488 |


MRR reported in the paper is: **0.04907**

On the chimera dataset:

| pre-trained model | L | Average RHO |
| --- | --- | --- |
| `wiki_all.sent.split.model` | L2 | 0.2945885566474934 |
| `wikidump.w2v.model` | L2 | 0.2417513120432906 |
| `wiki_all.sent.split.model` | L4 | 0.2934163091681338 |
| `wikidump.w2v.model` | L4 | 0.1597227113856652 |
| `wiki_all.sent.split.model` | L6 | 0.3529782502652243 |
| `wikidump.w2v.model` | L6 | 0.17972625545903392 |

in *sum_only* mode:

| pre-trained model | L | Average RHO |
| --- | --- | --- |
| `wiki_all.sent.split.model` | L2 | 0.33666460433580814 |
| `wikidump.w2v.model` | L2 | 0.2519347998127817 |
| `wiki_all.sent.split.model` | L4 | 0.36519180870706414 |
| `wikidump.w2v.model` | L4 | 0.1852681912170354 |
| `wiki_all.sent.split.model` | L6 | 0.40022239961777484 |
| `wikidump.w2v.model` | L6 | 0.2073600944244708 |

with the nonce2vec bugfix:

| pre-trained model | L | Average RHO |
| --- | --- | --- |
| `wiki_all.sent.split.model` | L2 |  |
| `wikidump.w2v.model` | L2 |  |
| `wiki_all.sent.split.model` | L4 |  |
| `wikidump.w2v.model` | L4 |  |
| `wiki_all.sent.split.model` | L6 |  |
| `wikidump.w2v.model` | L6 |  |

in *sum_only* mode:

| pre-trained model | L | Average RHO |
| --- | --- | --- |
| `wiki_all.sent.split.model` | L2 |  |
| `wikidump.w2v.model` | L2 |  |
| `wiki_all.sent.split.model` | L4 |  |
| `wikidump.w2v.model` | L4 |  |
| `wiki_all.sent.split.model` | L6 |  |
| `wikidump.w2v.model` | L6 |  |


Average RHO values reported in the paper are (for n2v):
- L2: 0.3320
- L4: 0.3668
- L6: 0.3890

Details:
- Results on the definitional dataset are robust across n2v versions and pre-trained w2v models
- Results on the chimera dataset are systematically lower than previously reported and we cannot replicate the hierarchy on L4. We find that the Sum version systematically outperforms n2v.

## Filtering XP

Twi big problems with nonce2vec:
1. random filter will return different context words at the build_vocab and training steps
2. nonce2vec relies on a re-initialization of the model each turn. Keeping learned vectors in memory will dramatically deteriorate perfs.

1. Context entropy (s2w)
2. Context word entropy (w2w)
3. Context information overlap (s2s)

On the nonce dataset

| model | no filter | random | self 20 | self 22 | w2w > 0 (CBOW) | w2w (BIDIR)
| --- | --- | --- | --- | --- | --- | --- | --- |
| sum-only | 0.01785 | 0.03327 | 0.02986 | 0.04816 | 0.04093 |  |
| nonce2vec | 0.03044 | 0.0539 | 0.04139 | 0.05860 | 0.04808 |  |

Why the improvement on random nonce2vec scores? Because we changed the
subsampling: originally there was two random subsampling: at sum time and
at train time. We changed so that the sum and training would occur on the
same context. This leads to an improvement in overall scores.

0.4093
On the chimera dataset:

| dataset | model | no filter | random | self 20 | self 22 | w2w > 0 (CBOW) | w2w (BIDIR)
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L2 | sum-only | 0.3079 | 0.3439 | 0.3329 | 0.3358 | 0.3235 | |
| L2 | nonce2vec | 0.2768 | 0.3322 | 0.2935 | 0.3421 | 0.3379 | |
| L4 | sum-only | 0.3624 | 0.3582 | 0.3679 | 0.3679 | 0.3491 | |
| L4 | nonce2vec | 0.3134 | 0.3515 | 0.3221 | 0.3466 | 0.3258 | |
| L6 | sum-only | 0.3812 | 0.3950 | 0.3789 | 0.4121 | 0.3907 | |
| L6 | nonce2vec | 0.3784 | 0.4052 | 0.3786 | 0.4124 | 0.3907 | |

Experiments when removing window decay and adjusting window size to all context for n2v

| model | random | self 22 | w2w > 0 (CBOW) |
| --- | --- | --- | --- |
| nonce2vec | 0.05262 | 0.05832 | 0.04807 |
No significant difference on nonces dataset

| dataset | model | random | self 22 | w2w > 0 (CBOW) |
| --- | --- | --- | --- | --- |
| L2 | nonce2vec | 0.3433 | 0.3421 | 0.3379 |
| L4 | nonce2vec | 0.3417 | 0.3466 | 0.3207 |
| L6 | nonce2vec | 0.4046 | 0.4124 | 0.3742 |
No significant difference either

Check average value and standard deviation of s2w and w2w with CBOW

Note that sample_decay does not work when not using random filter. This seems to
have a significant positive impact. Maybe implement a form of sample_decay.
Try all filters with a sample_decay

Experiments with a CBOW model trained on a larger window

Experiments when not resetting the model at each iteration: keep learned vectors in memory

Experiments on chimeras when sorting sentences by context entropy

| dataset | model | order | score |
| --- | --- | --- | --- |
| L2 | random | as-is | 0.3322 |
| L2 | random | asc   | 0.3391 |
| L2 | random | desc  | 0.3125 |
| L4 | random | as-is | 0.3515 |
| L4 | random | asc   | 0.3701 |
| L4 | random | desc  | 0.3619 |
| L6 | random | as-is | 0.4052 |
| L6 | random | asc   | 0.3902 |
| L6 | random | desc  | 0.3817 |

Experiments on chimeras when sorting by context entropy within margin of error
(round context entropy at the right decimal)

Spearman correlation = -0.18694416650898932
For s2w with rank on nonces

## Experiments:
Probabilities:
1. With CBOW to predict probabilities
2. With BIDIR LM to predict probabilities

S2W informativeness:
1. With the standard entropy
2. With a weighted entropy
3. With filtered context words

W2W informatieness:
1. With the 2 forms of S2W informativeness
2. With CBOW as a deviation with and without source context word
3. With BIDIR as a deviation on the neighboring words in the distribution

Testing S2W informativeness:
1. On the nonce dataset test correlation between score and informativeness
2. On the chimera dataset try sorting sentences by s2w informativeness

Testing W2W informativeness:
1. Sum on context words with W2W above specified threshold
2. Train nonce2vec with a dynamic window-size adjusted based on informative context words

Testing S2S informativeness:
1. On the chimera dataset, train on the most informative sentence only
