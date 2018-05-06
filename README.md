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
  --mode def_nonces \
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
  --mode def_nonces \
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
  --mode chimera \
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
| `wiki_all.sent.split` | 259376 | 0.7496 | Aurélie's wikidump |
| `wiki.all.utf8.sent.split.lower` | 308334 | 0.7085 | Alex's wikidump (lowercase UTF-8 version of Aurélie's) |

On the nonce dataset:

| pre-trained model | MRR |
| --- | --- |
| `wiki_all.sent.split` | 0.04879828330072024 |
| `wiki.all.utf8.sent.split.lower` | 0.030977350626280563 |

in *sum_only* mode:
| pre-trained model | MRR |
| --- | --- |
| `wiki_all.sent.split` |  |
| `wiki.all.utf8.sent.split.lower` |  |

MRR reported in the paper is: **0.04907**

On the chimera dataset:

| pre-trained model | L | Average RHO |
| --- | --- | --- |
| `wiki_all.sent.split` | L2 |  |
| `wiki.all.utf8.sent.split.lower` | L2 |  |
| `wiki_all.sent.split` | L4 |  |
| `wiki.all.utf8.sent.split.lower` | L4 |  |
| `wiki_all.sent.split` | L6 |  |
| `wiki.all.utf8.sent.split.lower` | L6 |  |

Average RHO values reported in the paper are:
- L2: 0.3320
- L4: 0.3668
- L6: 0.3890
