[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][travis-image]][travis-url]
[![MIT License][license-image]][license-url]
[![DOI][doi-image]][doi-url]

# nonce2vec
Welcome to Nonce2Vec!

This is the repo accompanying the paper [High-risk learning: acquiring new word
vectors from tiny data](http://aurelieherbelot.net/resources/papers/emnlp2017_final.pdf) (Herbelot &amp; Baroni, 2017). If you use this code,
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

**NEW!** We have now released v2 of Nonce2Vec which is packaged via pip and
runs on gensim v3.4.0. This should make it way easier for you to replicate
experiments.

## Install
```bash
pip3 install nonce2vec
```

## Download and extract the required resources
To download the definitional, chimeras and MEN datasets:
```bash
wget http://129.194.21.122/~kabbach/noncedef.chimeras.men.7z
```
To use the pretrained gensim model from Herbelot and Baroni (2017):
```bash
wget http://129.194.21.122/~kabbach/wiki_all.sent.split.model.7z
```

## Generate a pre-trained word2vec model
If you want to generate a new gensim.word2vec model from scratch and do not want to rely on the `wiki_all.sent.split.model`:

### Download/Generate a Wikipedia dump
To use the same Wikipedia dump as Herbelot and Baroni (2017):
```bash
wget http://129.194.21.122/~kabbach/wiki.all.utf8.sent.split.lower.7z
```

Else, to create a new Wikipedia dump from an different archive, check out
[WiToKit](https://github.com/akb89/witokit).

### Train the background model
You can train Word2Vec with gensim via the nonce2vec package:

```bash
n2v train \
  --data /absolute/path/to/wikipedia/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use
  --train-mode skipgram
```

### Check the correlation with the MEN dataset
```bash
n2v check \
  --data /absolute/path/to/MEN/MEN_dataset_natural_form_full
  --model /absolute/path/to/gensim/word2vec/model
```

## Replication

### Test nonce2vec on the nonce definitional dataset
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
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5
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
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5
```

### Results
Results on nonce2vec v2.x are slightly different than those reported to in the
original EMNLP paper due to several bugfix in how gensim originally
handled subsampling with `random.rand()`.

| DATASET  | MRR / RHO |
| --- | --- |
| Definitional | 0.04846 |
| Chimeras L2 | 0.3407 |
| Chimeras L4 | 0.3457 |
| Chimeras L6 | 0.4001 |

[release-image]:https://img.shields.io/github/release/minimalparts/nonce2vec.svg?style=flat-square
[release-url]:https://github.com/minimalparts/nonce2vec/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/nonce2vec.svg?style=flat-square
[pypi-url]:https://pypi.org/project/nonce2vec/
[travis-image]:https://img.shields.io/travis/minimalparts/nonce2vec.svg?style=flat-square
[travis-url]:https://travis-ci.org/minimalparts/nonce2vec
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt
[doi-image]:https://img.shields.io/badge/DOI-10.5281%2Fzenodo.1423290-blue.svg?style=flat-square
[doi-url]:https://zenodo.org/badge/latestdoi/96074751
