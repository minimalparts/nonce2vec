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


Average RHO values reported in the paper are (for n2v):
- L2: 0.3320
- L4: 0.3668
- L6: 0.3890

Details:
- Results on the definitional dataset are robust across n2v versions and pre-trained w2v models
- Results on the chimera dataset are systematically lower than previously reported and we cannot replicate the hierarchy on L4. We find that the Sum version systematically outperforms n2v. 
