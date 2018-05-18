# Nonce2Vec

## Proposal
We propose to quantify how informative a given context is regarding a target
word. Our proposal is articulated around two measures:
1. Context entropy: which measures how 'informative' a given set of context
words is regarding a given target.
2. Context word entropy: which measures how 'informative' a given context
word is regarding a given target.

Both measures make use of Shannon entropy, relying on the intuition that
an informative context is a context specific to a given word or limited
set of words, and that an informative context is therefore a context with
a *low* entropy.

Formally, context entropy CE is defined as:
```
CE(context) = 1 - (shannon_entropy(target|context) / log(len(vocab)))
```
Where the shannon entropy is computed over the vocabulary with CBOW, which
outputs a probability distribution over words in the vocab given a set of
context words. CE has values is [0, 1], 1 being a maximally informative context

Formally, context word entropy CWE is defined as:
CWE(context_words) = CE(context_with_context_word) - CE(context_without_context_word)

CWE has values in [-1, 1], 1 being a word that is maximally informative in
the given context.

## Experiments

### Replication
We first start by trying to replicate (H & B 2017) results from scratch, that is,
by retraining the background model and running their code as-is with the
same hyperparameters.

Important note: we fixed the learning rate bug that would reset the learning
rate to its original value at each sentence iteration.

Results are:
| #XP | test set | eval metric |
| --- | --- | --- |
| 001 | nonces |  |
| 002 | chimeras L2 |  |
| 003 | chimeras L4 |  |
| 004 | chimeras L6 |  |

Conclusion:

### Filtering and summing
In the first round of experiment we start from the following considerations:
- The additive baseline in (H & B 2017) summed over all in-vocab context words filtered with a stopwords list
- In N2V, the sum is performed over a set of context words (in vocab) with a random subsampling filter

We ask the following questions:
- What is the impact of removing duplicates (summing over a list or a set of context words) ?
- What is the impact of filtering context words ?

We test on both the nonces and chimeras datasets with the following configurations:
1. no filtering of context words
2. a random subsampling filter (as in n2v)
3. a self-information filter which simulates a stopwords list based on word frequency in a large corpus
4. a context word entropy-based filter which only keeps 'informative' context words (cwe > 0)

All filters are tested in 'sum-only' mode, with and without duplicates
(summing over list vs. summing over set)

Results are:

The baseline we are trying to beat is:
| nonces | chimeras L2 | chimeras L4 | chimeras L6 |
| --- | --- | --- | --- | --- |
| | | | | |
| | | | | |

Details:
| #XP | test set | sum over | no filter | random | self | cwe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 005 | nonces | list | 0.01955 | 0.02600 | 0.02762 | 0.03284 |
| 006 | nonces | set |  |  |  |  |
| 007 | chimeras L2 | list |  |  |  |  |
| 008 | chimeras L2 | set |  |  |  |  |
| 009 | chimeras L4 | list |  |  |  |  |
| 010 | chimeras L4 | set |  |  |  |  |
| 011 | chimeras L6 | list |  |  |  |  |
| 012 | chimeras L6 | set |  |  |  |  |

### Nonce2Vec, exponential decay, filtering and sorting
We now focus on measuring the contribution of n2v 'training' of context words,
performed after nonce vectors have been initialized via summing over
context word vectors.

We are interested in:
- quantifying the impact of filtering training context words with cwe vs.
applying a random filter with sample (and window) decay.
- checking whether or not sorting context words by cwe helps training

All the following models rely on initialization based on summing over
a set/list (?) of context words filtered by their respective cwe.

We compare 3 models:
- n2v-asi: the original n2v model, random filter + window decay + sample decay (minus the original bug)
- n2v-cwe: a model using exponential decay, no window-size limit (window = whole context) and a cwe filter
- n2v-sort: the n2v-cwe model sorting context words in descending order of cwe value

Results are:

Details:

### cwe-based learning rate
