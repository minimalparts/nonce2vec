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

Evaluation metric is:
- MRR for nonces
- Average RHO for chimeras

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
- The additive baseline in (H & B 2017) summed over all in-vocab context words
filtered with a stopwords list
- In N2V, the sum is performed over a set of context words (in vocab) with a
random subsampling filter

We ask the following questions:
- What is the impact of removing duplicates (summing over a list or a set of
context words) ?
- What is the impact of filtering context words ?

We test on both the nonces and chimeras datasets with the following configurations:
1. no filtering of context words
2. a random subsampling filter (as in n2v)
3. a self-information filter which simulates a stopwords list based on word
frequency in a large corpus
4. a context word entropy-based filter which only keeps 'informative' context
words (cwe > 0)

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
- n2v-asi: the original n2v model, random filter + window decay + sample decay
(minus the original bug)
- n2v-cwe: a model using exponential decay, no window-size limit
(window = whole context) and a cwe filter
- n2v-sort: the n2v-cwe model sorting context words in descending order
of cwe value

Reminder: the learning rate alpha is computed by:
```python
def compute_exp_alpha(nonce_count, lambda_den, alpha, min_alpha):
    exp_decay = -(nonce_count-1) / lambda_den
    if alpha * exp(exp_decay) > min_alpha:
        return alpha * exp(exp_decay)
    return min_alpha
```

Results are:

Best system on each:
| #XP | test set  | setup | train over | eval metric |
| --- | --- | --- | --- | --- |
| 0 | nonces |  |  |  |
| 0 | chimeras L2 |  |  |  |  |  |
| 0 | chimeras L4 |  |  |  |  |  |
| 0 | chimeras L6 |  |  |  |  |  |

Details:

| #XP | test set  | setup | train over | eval metric |
| --- | --- | --- | --- | --- |
| 013 | nonces | n2v-asi | list |  |
| 014 | nonces | n2v-cwe | list |  |  |  |
| 015 | nonces | n2v-sort | list |  |  |  |
| 016 | nonces | n2v-sort | set |  |  |  |
| 017 | L2 | n2v-asi | list |  |  |  |
| 018 | L2 | n2v-cwe | list |  |  |  |
| 019 | L2 | n2v-sort | list |  |  |  |
| 020 | L2 | n2v-sort | set |  |  |  |
| 021 | L4 | n2v-asi | list |  |  |  |
| 022 | L4 | n2v-cwe | list |  |  |  |
| 023 | L4 | n2v-sort | list |  |  |  |
| 024 | L4 | n2v-sort | set |  |  |  |
| 025 | L6 | n2v-asi | list |  |  |  |
| 026 | L6 | n2v-cwe | list |  |  |  |
| 027 | L6 | n2v-sort | list |  |  |  |
| 028 | L6 | n2v-sort | set |  |  |  |

### cwe-based learning rate
We propose an alternative to the exponential decay-based learning rate.
We compute the learning rate for each context-target word pair by:

```python
def compute_cwe_alpha(cwe, kappa, beta, alpha, min_alpha):
    x = tanh(cwe*beta)
    decay = (exp(kappa*(x+1)) - 1) / (exp(2*kappa) - 1)
    if decay * alpha > min_alpha:
        return decay * alpha
    return min_alpha
```

Where
- `cwe`is the context word entropy of the context word
- `kappa` is a hyperparameter controlling how much to downgrade low-cwe words
- `beta` is a hyperparameter controlling the bias of the cwe parameter
- `alpha` is the initial learning rate
- `min_alpha` is the minimal learning rate

In short, the learning rate will be a function of the context word entropy.
The `tanh` function is here to compensate for the cwe values being too
centered around 0: we use the beta parameter to 'rescale' the cwe values
in [-1,1] to get more discriminant learning rates across cwe values.

The `cwe_alpha` function will exponentially decrease the learning rate as
cwe decreases. The 'exponentiality' of this decrease is controlled by kappa.

The intuition is that we want the model to replace a context filter by being
able to determine by itself, for each context word, how much it should learn
from it and rescaling the learning rate accordingly.

We test with values:
- alpha: 1, 3, 5, 10
- beta: 100, 500, 1000, 1500, 2000
- kappa: 1, 2, 3

Our best results are obtained with alpha = , beta = , kappa = .

We are trying to beat the same model with a cwe train filter 

Details:

| #XP | alpha | beta | kappa | train over | nonces | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 029 | 1 | 1000 | 1 | list | | | | |
| 030 | 1 | 1000 | 1 | set | | | | |
| 031 | 1 | 1000 | 2 | list | | | | |
| 032 | 1 | 1000 | 2 | set | | | | |
| 033 | 1 | 500 | 2 | list | | | | |
| 034 | 1 | 500 | 2 | set | | | | |
| 035 | 3 | 500 | 2 | list | | | | |
| 036 | 3 | 500 | 2 | set | | | | |
| 037 | 5 | 500 | 2 | list | | | | |
| 038 | 5 | 500 | 2 | set | | | | |
| 0 |  |  |  | set | | | | |

### Quality of the background model
The performance of the cwe-based approach (filter and learning rate) are
conditioned on the 'quality' of the background model probability distributions.
Here we question whether hyperparameters of the background model
(window size, subsampling rate, min count)
positively impact cwe-based learning.

| #XP | window | sample | min  | nonces | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 |  |  |  |  | | | |
| 0 |  |  |  |  | | | |
| 0 |  |  |  |  | | | |
| 0 |  |  |  |  | | | |
| 0 |  |  |  |  | | | |
| 0 |  |  |  |  | | | |


### Going further: is context entropy useful in itself?
Two questions I would like to ask if I still have time:
1. Can context entropy help us determine 'minimally-informative' sentences?
2. Does context entropy correlates well with our overall results?
