# nonce2vec
[![GitHub release][release-image]][release-url]
[![PyPI release][pypi-image]][pypi-url]
[![Build][travis-image]][travis-url]
[![MIT License][license-image]][license-url]

Welcome to Nonce2Vec!

The main branch of this repository now refers to the Kabbach et al. (2019) ACL SRW 2019 paper [*Towards incremental learning of word embeddings using context informativeness*](https://www.aclweb.org/anthology/P19-2022).

**If you are looking for the Herbelot and Baroni (2017) repository, check out the [emnlp2017](https://github.com/minimalparts/nonce2vec/tree/release/emnlp2017) branch.**

If you use this code, please cite:
```tex
@inproceedings{kabbach-etal-2019-towards,
    title = "Towards Incremental Learning of Word Embeddings Using Context Informativeness",
    author = "Kabbach, Alexandre  and
      Gulordava, Kristina  and
      Herbelot, Aur{\'e}lie",
    booktitle = "Proceedings of the 57th Conference of the Association for Computational Linguistics: Student Research Workshop",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-2022",
    pages = "162--168"
}
```

**Abstract**

*In this paper, we investigate the task of learning word embeddings from very sparse data in an incremental, cognitively-plausible way. We focus on the notion of informativeness, that is, the idea that some content is more valuable to the learning process than other. We further highlight the challenges of online learning and argue that previous systems fall short of implementing incrementality. Concretely, we incorporate informativeness in a previously proposed model of nonce learning, using it for context selection and learning rate modulation. We test our system on the task of learning new words from definitions, as well as on the task of learning new words from potentially uninformative contexts. We demonstrate that informativeness is crucial to obtaining state-of-the-art performance in a truly incremental setup.*

## A note on the code
We have significantly refactored the original Nonce2Vec code in order to make replication easier and to make it work with gensim v3.x. You can use Nonce2Vec v2.x to replicate the results of the SRW paper. However, to replicate results of the original ENMLP paper, refer to Nonce2Vec v1.x found under the [emnlp2017 branch](https://github.com/minimalparts/nonce2vec/tree/release/emnlp2017) as we **cannot** guarantee fair replication between v1.x and v2.x.

## Install
You can install Nonce2Vec via pip:
```bash
pip3 install nonce2vec
```
or, after a git clone, via:
```bash
python3 setup.py install
```

## Pre-requisites
To run Nonce2Vec, you need two gensim Word2Vec models (a skipgram model and a cbow model to compute informativeness-metrics). You can download the skipgram model from:
```bash
wget backup.3azouz.net/gensim.w2v.skipgram.model.7z
```
and the cbow model from:
```sh
wget backup.3azouz.net/gensim.w2v.cbow.model.7z
```
or generate both yourself following the instructions below.

### Generating a Word2Vec model from a Wikipedia dump
You can download our English Wikipedia dump of January 2019 here:
```bash
wget backup.3azouz.net/enwiki.20190120.7z
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

To check the correlation of your word2vec model(s) with the MEN dataset, run:
```bash
n2v check-men \
  ---model /absolute/path/to/gensim/w2v/model
```

## Running the code
Running Nonce2Vec on the definitional of chimeras datasets is done via the `n2v test` command. You can pass in the `--reload` parameter to run in `one-shot` mode, without it the code runs in incremental model by default. You can further pass in the `--shuffle` parameter to shuffle the test set before running n2v.

You will find below a list of commands corresponding to the experiments reported in the SRW 2019 paper. For example, to test the SUM CWI model (a basic sum model with context-word-informativeness-based filtering), which provides a rather robust baseline on all datasets in incremental setup, run, for the definitional dataset:
```bash
n2v test \
  --on def \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --info-model /absolute/path/to/gensim/w2v/cbow/model \
  --sum-only \
  --sum-filter cwi \
  --sum-threshold 0
```

To run the N2V CWI alpha model on the chimera L4 test set, with shuffling and in
one-shot evaluation setup (which provides SOTA performance), do:
```bash
n2v test \
  --on l4 \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --info-model /absolute/path/to/gensim/w2v/cbow/model \
  --sum-filter cwi \
  --sum-threshold 0 \
  --train-with cwi_alpha \
  --alpha 1.0 \
  --beta 1000 \
  --kappa 1 \
  --neg 3 \
  --epochs 1 \
  --reload
```

To test N2V as-is (the original N2V code without background freezing), in incremental setup on the definitional dataset, do:
```bash
n2v test \
  --on def \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```

To test N2V CWI init (the original N2V with CWI-based sum initialization) on the definitional dataset in one-shot evaluation setup, do:
```bash
n2v test \
  --on def \
  --model /absolute/path/to/gensim/w2v/skipgram/model \
  --info-model /absolute/path/to/gensim/w2v/cbow/model \
  --sum-filter cwi \
  --sum-threshold 0 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication \
  --reload
```


[release-image]:https://img.shields.io/github/release/minimalparts/nonce2vec.svg?style=flat-square
[release-url]:https://github.com/minimalparts/nonce2vec/releases/latest
[pypi-image]:https://img.shields.io/pypi/v/nonce2vec.svg?style=flat-square
[pypi-url]:https://pypi.org/project/nonce2vec/
[travis-image]:https://img.shields.io/travis/akb89/nonce2vec.svg?style=flat-square
[travis-url]:https://travis-ci.org/akb89/nonce2vec
[license-image]:http://img.shields.io/badge/license-MIT-000000.svg?style=flat-square
[license-url]:LICENSE.txt
