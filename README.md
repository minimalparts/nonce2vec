[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][travis-image]][travis-url]
[![MIT License][license-image]][license-url]

# nonce2vec
Welcome to Nonce2Vec!

The main branch of this repository now refers to the Kabbach et al. (2019) ACL SRW 2019 paper *Towards incremental learning of word embeddings using context informativeness*.

**If you are looking for the Herbelot and Baroni (2017) repository, check out the [emnlp2017 branch](https://github.com/minimalparts/nonce2vec/tree/release/emnlp2017)**

If you use this code, please cite:
```tex
@InProceedings{kabbachetal2019,
  author    = {Kabbach, Alexandre and Gulordava, Kristina and Herbelot, Aur\'{e}lie},
  title     = {Towards incremental learning of word embeddings using context informativeness},
  booktitle = {},
  month     = {August},
  year      = {2019},
  address   = {Florence, Italy},
  publisher = {Association for Computational Linguistics},
  pages     = {},
  url       = {}
}
```

**Abstract**

*In this paper, we investigate the task of learning word embeddings from very sparse data in an incremental, cognitively-plausible way. We focus on the notion of informativeness, that is, the idea that some content is more valuable to the learning process than other. We further highlight the challenges of online learning and argue that previous systems fall short of implementing incrementality. Concretely, we incorporate informativeness in a previously proposed model of nonce learning, using it for context selection and learning rate modulation. We test our system on the task of learning new words from definitions, as well as on the task of learning new words from potentially uninformative contexts. We demonstrate that infor- mativeness is crucial to obtaining state-of-the-art performance in a truly incremental setup.*

## A note on the code
We have significantly refactored the original Nonce2Vec code in order to make replication easier and to make it work with gensim v3.x. You can use Nonce2Vec v2.x to replicate the results of the SRW paper. However, to replicate results of the original ENMLP paper, refer to Nonce2Vec v1.x found under the [emnlp2017 branch](https://github.com/minimalparts/nonce2vec/tree/release/emnlp2017) as we **cannot** guarantee fair replication between v1.x and v2.x.

## Install
You can install Nonce2Vec via:
```bash
pip3 install nonce2vec
```

## Pre-requisites
To run Nonce2Vec, you need two gensim Word2Vec models (a skipgram model and a cbow model to compute informativeness-metrics). You can download the skipgram model from:
```bash
wget http://129.194.21.122/~kabbach/gensim.w2v.skipgram.model.7z
```
and the cbow model from:
```sh
wget http://129.194.21.122/~kabbach/gensim.w2v.cbow.model.7z
```
or generate both yourself following the instructions below.

### Generating a Word2Vec model from a Wikipedia dump
You can download our English Wikipedia dump of January 2019 here:
```bash
wget http://129.194.21.122/~kabbach/enwiki.20190120.7z
```
If you want to generate a completely new (tokenized-one-sentence-per-line) dump
of Wikipedia, for English or any other language, check out [WiToKit](https://github.com/akb89/witokit).

Once you have a Wikipedia txt dump, you can generate a gensim Word2Vec skipgram model via:
```bash
n2v train \
  --data /absolute/path/to/wikipedia/tokenized/text/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use \
  --train-mode skipgram
```
and a gensim Word2Vec cbow model via:
```bash
n2v train \
  --data /absolute/path/to/wikipedia/tokenized/text/dump \
  --outputdir /absolute/path/to/dir/where/to/store/w2v/model \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads number_of_cpu_threads_to_use \
  --train-mode cbow
```

## Running the code


[release-image]:https://img.shields.io/github/release/minimalparts/nonce2vec.svg?style=flat-square
[release-url]:https://github.com/minimalparts/nonce2vec/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/nonce2vec.svg?style=flat-square
[pypi-url]:https://pypi.org/project/nonce2vec/
[travis-image]:https://img.shields.io/travis/akb89/nonce2vec.svg?style=flat-square
[travis-url]:https://travis-ci.org/akb89/nonce2vec
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt
