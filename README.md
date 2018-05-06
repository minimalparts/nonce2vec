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

```

### Test nonce2vec on the chimera dataset
```bash

```


## A note on the code
We have had queries about *where* exactly the Nonce2Vec code resides. Since it is a modification of the original gensim Word2Vec model, it is located in the gensim/models directory, confusingly still under the name *word2vec.py*. All modifications described in the paper are implemented in that file. Note that there is no C implementation of Nonce2Vec, so the program runs on standard numpy. Also, only skipgram is implemented -- the cbow functions in the code are original Word2Vec.


## Pre-requisites
You will need a pre-trained gensim model. You can go and train one yourself, using the gensim repo at [https://github.com/rare-technologies/gensim](https://github.com/rare-technologies/gensim), or simply download ours, pre-trained on Wikipedia:

`wget http://clic.cimec.unitn.it/~aurelie.herbelot/wiki_all.model.tar.gz`

If you use our tar file, the content should be unpacked into the models/ directory of the repo.

## Running the code

Here is an example of how to run the code on the test set of the definitional dataset, with the best identified parameters from the paper:

`python test_def_nonces.py models/wiki_all.sent.split.model data/definitions/nonce.definitions.300.test 1 10000 3 15 1 70 1.9 5`

For the chimeras dataset, you can run with:

`python test_chimeras.py models/wiki_all.sent.split.model data/chimeras/chimeras.dataset.l4.tokenised.test.txt 1 10000 3 15 1 70 1.9 5`

(changing the chimeras test set for testing on 2, 4 or 6 sentences).


## The data

In the data/ folder, you will find two datasets, split into training and test sets:

* The Wikipedia 'definitional dataset', produced specifically for this paper.
* A pre-processed version of the 'Chimera dataset' (Lazaridou et al, 2017). More details on this data are to be found in the README of the data/chimeras/ directory.

We thank the authors of the Chimera dataset for letting us use their data. We direct users to the original paper:

A. Lazaridou, M. Marelli and M. Baroni. 2017. Multimodal word meaning induction from minimal exposure to natural text. *Cognitive Science*. 41(S4): 677-705.
