[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][travis-image]][travis-url]
[![MIT License][license-image]][license-url]
[![DOI](https://zenodo.org/badge/96074751.svg)](https://zenodo.org/badge/latestdoi/96074751)

# nonce2vec
Welcome to Nonce2Vec!

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

**NEW** We have now released v2.0 of Nonce2Vec which is packaged via pip and
runs on gensim

## Install
```bash
pip3 install nonce2vec
```

## What's new

## Download and extract the required resources
To download the nonces and chimeras datasets:
```bash
wget http://129.194.21.122/~kabbach/noncedef.chimeras.men.7z
```
To use the pretrained gensim model from Herbelot and Baroni (2017):
```bash
wget http://129.194.21.122/~kabbach/wiki_all.model.7z
```

## Generate a pre-trained word2vec model
To generate a gensim.word2vec model from scratch, with the same wikidump
and the same hyperparameters as Herbelot and Baroni (2017):

### Download the Wikipedia dump
Lowercase UTF-8 dump:
```bash
wget http://129.194.21.122/~kabbach/wiki.all.utf8.sent.split.lower.7z
```

### Train the background model
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
  --train_mode skipgram
```

### Check the correlation with the MEN dataset
```bash
n2v check \
  --data /absolute/path/to/MEN/MEN_dataset_natural_form_full
  --model /absolute/path/to/gensim/word2vec/model
```

## Test nonce2vec on the nonce definitional dataset
```bash
n2v test \
  --on nonces \
  --model /absolute/path/to/pretrained/w2v/model \
  --data /absolute/path/to/nonce.definitions.299.test \
  --alpha 1 \
  --neg 3 \
  --window 15 \
  --sample 10000 \
  --epochs 1 \
  --min_count 1 \
  --lambda 70 \
  --sample_decay 1.9 \
  --window_decay 5 \
  --sum_filter random \
  --sum_over_set \
  --replication
```


### Test nonce2vec on the chimeras dataset
```bash
n2v test \
  --on chimeras \
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
  --sum_filter random \
  --sum_over_set \
  --replication
```

[release-image]:https://img.shields.io/github/release/minimalparts/nonce2vec.svg?style=flat-square
[release-url]:https://github.com/minimalparts/nonce2vec/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/nonce2vec.svg?style=flat-square
[pypi-url]:https://pypi.org/project/nonce2vec/
[travis-image]:https://img.shields.io/travis/minimalparts/nonce2vec.svg?style=flat-square
[travis-url]:https://travis-ci.org/minimalparts/nonce2vec
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt
