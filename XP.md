# Experiments

## Subsampling
We try and experiment with different kind of subsampling in *sum-only* mode on
the nonce definitional dataset.
1. Random subsampling
2. Self information with threshold at 22 (explain why)
3. Context informativeness > 0 (on a small window model and large window model)

```bash
n2v test --what performances --on nonces --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.300.test --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --sum_only --filter cwe --threshold 0
```

Why 22? subsampling in w2v leads to:
```
sentence: [['romsey', 'is', 'a', 'small', 'market', 'town', 'in', 'the', 'county', 'of', 'hampshire', 'england']]
word = is | log(sample_int) = 21.272400549808868
word = county | log(sample_int) = 22.18070977791825
word = small | log(sample_int) = 22.18070977791825
word = england | log(sample_int) = 22.18070977791825
word = in | log(sample_int) = 20.663636750919736
word = a | log(sample_int) = 20.845713629907426
word = hampshire | log(sample_int) = 22.18070977791825
word = the | log(sample_int) = 20.141004258007644
word = town | log(sample_int) = 22.18070977791825
word = market | log(sample_int) = 22.18070977791825
word = of | log(sample_int) = 20.565777364047005
```
on average content word will be above 22

| | no filter | random | self 22 | CE > 0 |
| --- | --- | --- | --- | --- |
| MRR | 0.02024 | 0.03088 | 0.03067 |  |
| SPR | -0.21 |  | -0.17 |  |

Impact of window size of pretrained model:

| window size | MRR |
| --- | --- |
| 5 |  |
| 10 |  |
| 15 |  |
| 20 | 0.03732608 |
| 50 |  |

Problem with replication: the order of words matter when training on nonce2vec!! This totally depends on the underlying vocabulary!

## Sorting words by CWE when training with nonce2vec

First replicate subsampling when training with nonce2vec
On important point: in the original nonce2vec code, sum is operated on a different context than train due to the random
subsampling beeing operated twice with a different RandomState
Also, sum is done on a set of context words when training is done on a list (not removing duplicates)

Bugfixes:
1. With the alpha learning rate
2. With the min_count (specified after building vocab)

First XP: sorting words when training (with exp decay)

Second XP: re-design exp decay to be function of cwe


n2v as-is

| no filter | random | self 22 | CE > 0 |
| --- | --- | --- | --- |
| 0.02177 | 0.02992 | 0.03026 |  |

n2v window size as len(context) no sort

| no filter | random | self 22 | CE > 0 |
| --- | --- | --- | --- |
| 0.02178 | 0.03098 | 0.03034 |  |

n2v big window sort cwe desc

| no filter | random | self 22 | CE > 0 |
| --- | --- | --- | --- |
|  |  |  |  |

n2v big window sort cwe asc

| no filter | random | self 22 | CE > 0 |
| --- | --- | --- | --- |
|  |  |  |  |
