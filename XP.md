# Experiments

Follow-up of the N2V meeting of 14.12.2018

## Problem
Originally (in an old version of the N2V code) it was observed that the
results would dramatically decrease when not reinitializing N2V at each
iteration (on the def. dataset, each iteration being a nonce).

Results actually suggested that all 'predicted' words ended up in the
same region upon summing over context.

(Another effect of 'sum' noted is that summing over a set or a list of
context words changed results rather significantly)

## XP1
Do we still observe this phenomenon on the latest version of N2V?
(add option to reinitialize space at each iteration)

## XP2
Does this effect persists upon filtering the summed over context?
1. Randomly initialize vectors (rather than summing)
2. Sum over a random set of words (selected in the vocabulary)
3. Sum over a uniform-distrib-based sampled set of words
4. Sum over a set of ctx words filtered based on informativeness

## Comments
If the effect of sum is confirmed, this would suggest that all words in
every context have a particular configuration in the space.
Could we *quantify*/*characterize* this configuration? By, e.g., measuring
the cosine sim. between each word in the context, by having a measure of
the absolute position of each context word relative to one another.


## Running experiments

XP001
```shell
n2v test --on definitions --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum-filter random --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --reload
```
MRR = 0.04303

XP101
```shell
n2v test --on definitions --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum-filter random --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --reload
```
MRR = 0.04400

XP002
```
n2v test --on definitions --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum-filter random --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication
```
MRR = 0.01076

XP102
```
n2v test --on definitions --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum-filter random --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication
```
MRR = 0.01793

XP003
```
n2v test --on definitions --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --sample 10000 --sum-filter self --sum-threshold 22
```
MRR = 0.01929

XP103
```
n2v test --on definitions --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --sample 10000 --sum-filter self --sum-threshold 22
```
MRR = 0.02229

XP004
```
n2v test --on definitions --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info-model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --sample 10000 --sum-filter cwi --sum-threshold 0
```
MRR = 0.02650

XP005
```
n2v test --on definitions --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info-model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --sample 10000 --sum-filter cwi --sum-threshold 0 --weighted --beta 1000
```
MRR = 0.02007

XP006
```
n2v test --on definitions --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info-model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --sample 10000 --sum-filter cwi --sum-threshold 0 --reload
```
MRR = 0.045264

Conclusions:
Without reloading the background model at each iteration, we observe a
decrease in performance (from 0.04400 to 0.01793).
With filtering, results improve from 0.01793 to 0.02229 and 0.02650 (with cwi)

## Chimeras

```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --sum-filter random --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication
```

Replication RHO (on wiki_all L2) = 0.33362

```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --sum-filter random --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --reduced
```

XP with summing over the context of all sentences

| #XP | L | RHO |
| --- | --- | --- |
| 007 | L2 | 0.29816 |
| 008 | L4 | 0.36747 |
| 009 | L6 | 0.36600 |

XP with summing over the context of the first sentence only

| #XP | L | RHO |
| --- | --- | --- |
| 010 | L2 | 0.28157 |
| 011 | L4 | 0.2928 |
| 012 | L6 | 0.3211 |

XP with summing over the context of all sentences with cwi
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info-model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --sum-filter cwi --sum-threshold 0
```
| #XP | L | RHO |
| --- | --- | --- |
| 013 | L2 | 0.317142 |
| 014 | L4 | 0.3586048 |
| 015 | L6 |  |

XP with summing over the context of the first sentence with cwi
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info-model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --sample 10000 --alpha 1.0 --neg 3 --window 15 --epochs 1 --lambda 70 --sample-decay 1.9 --window-decay 5 --replication --sum-filter cwi --sum-threshold 0 --reduced
```
| #XP | L | RHO |
| --- | --- | --- |
| 016 | L2 | 0.28259 |
| 017 | L4 | 0.27882 |
| 018 | L6 | 0.29566 |


## Testing MRR on definitions between "gold" background models

| #XP | MODEL-1 | MODEL-2 | MRR |
| --- | --- | --- | --- |
| 019 | SG1 | CBOW | |
| 020 | CBOW | SG1 | |
| 021 | SG1 | SG2 | |
| 022 | SG2 | SG1 | |
