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
```
CWE(context_words) = CE(context_with_context_word) - CE(context_without_context_word)
```

CWE has values in [-1, 1], 1 being a word that is maximally informative in
the given context.

## Summary of the results

On replicating past n2v results:
- we could not replicate the quality of the background model with previously
reported hyperparameters (see MEN score, we get a lower quality model)
- the reported sum baseline was actually a no-filter-sum-over-list baseline.
When comparing to that baseline, we found n2v to consistently outperform
the baseline on all datasets; however
- we found the performance of n2v on nonces to be lower (in absolute)
- we found the performance of n2v on chimeras to be higher (in comparison)
- we found the contribution of n2v to mostly come from the random filter in the
initialization phase, in comparison with the no-filter-sum-over-list baseline.

Upon proposing a context-word-informativeness filter:
- we found the additive model based on sum over cwi filtered set of context words
to outperform all alternative additive baselines by a significant margin
(those baselines included no filter, random, self; both in list and set config)
- we found n2v as-is to still outperform the strong cwi sum baseline (overall),
when initialized on cwi sum. We therefore found the exponential decay training
model to consistently positively impact results (except on L6)

Upon proposing an alternative exponential decay-based learning model
sorting context words by cwi in descending order:
- we found mixed results: overall this model beats the additive baseline, but
only marginally, and is often outperformed by n2v-asis.

Upon proposing an exponential cwi learning model:
- awaiting xp

Upon testing the robustness of our results against H&B's background model:
- awaiting xp

Upong testing the correlation between density and MRR scores:
- awaiting xp

The most important contribution of this work is probably the CWI additive
baseline which proved very hard to beat.

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

Results are (with our background model):

| #XP | test set | eval metric |
| --- | --- | --- |
| 001 | nonces | 0.02954 |
| 002 | chimeras L2 | 0.2049  |
| 003 | chimeras L4 | 0.1895|
| 004 | chimeras L6 | 0.1844 |

With H & B's background model:

| #XP | test set | eval metric |
| --- | --- | --- |
| 101 | nonces | 0.04303 |
| 102 | chimeras L2 | 0.3336 |
| 103 | chimeras L4 | 0.3438 |
| 104 | chimeras L6 | 0.3922 |

Differences of quality between the two background model (correlation with MEN):
- our own: 0.7032
- H & B: 0.7496

Conclusion:
- we cannot replicate the MEN correlation by training a background model with
the same hyperparameters as H & B on gensim 3.4 (they used 0.13)
- a drop in background model quality (measured by MEN correlation) seems to lead
to a significant drop in performance of N2V, especially on the definitional dataset
- with our background model, we see a marginal difference between N2V and
an additive model with a stopwords list (see 'self' mode in XP section below)
- the improvements observed (by N2V over SUM) seem to be due to N2V summing
over a *set* of context words in the initialization phase.

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

**RESULT**
An additive model (SUM) based on a CWE-filtered set of context words
systematically outperforms any other additive model and provides a robust
baseline across all datasets.


The baseline we are trying to beat is:

| nonces | chimeras L2 | chimeras L4 | chimeras L6 |
| --- | --- | --- | --- |
| 0.03655 | 0.2759 | 0.2400 | 0.3312 |

Details:

| #XP | test set | sum over | no filter | random | self | cwe |
| --- | --- | --- | --- | --- | --- | --- |
| 005 | nonces | list | 0.01955 | 0.02600 | 0.02762 | 0.03284 |
| 006 | nonces | set | 0.02031 | 0.02743 | 0.03077 | 0.03655 |
| 007 | chimeras L2 | list | 0.1832 | 0.2115 | 0.2157 | 0.2557 |
| 008 | chimeras L2 | set | 0.1966 | 0.1995 | 0.2213 | 0.2759 |
| 009 | chimeras L4 | list | 0.1367 | 0.1722 | 0.1780 | 0.2383 |
| 010 | chimeras L4 | set | 0.1737 | 0.1879 | 0.1785 | 0.2400 |
| 011 | chimeras L6 | list | 0.1290 | 0.1616 | 0.1999 | 0.3312 |
| 012 | chimeras L6 | set | 0.1652 | 0.1800 | 0.2113 | 0.3256 |

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
(window = whole context) and a cwe filter on training context
- n2v-sort: the n2v-cwe model sorting context words in descending order
of cwe value

We do not sum over a set with n2v-asi as we assume the window/window_decay
part of the model to deal with this.

Reminder: the learning rate alpha is computed by:
```python
def compute_exp_alpha(nonce_count, lambda_den, alpha, min_alpha):
    exp_decay = -(nonce_count-1) / lambda_den
    if alpha * exp(exp_decay) > min_alpha:
        return alpha * exp(exp_decay)
    return min_alpha
```

Results are:

The baseline we are trying to beat is:

| nonces | chimeras L2 | chimeras L4 | chimeras L6 |
| --- | --- | --- | --- |
| 0.03655 | 0.2759 | 0.2400 | 0.3312 |

Best system on each:

| #XP | test set  | setup | train over | eval metric |
| --- | --- | --- | --- | --- |
| 013 | nonces | n2v-asi | list | 0.03698 |
| 018 | chimeras L2 | n2v-asi | list | 0.2786 |
| 026 | chimeras L4 | n2v-sort | list | 0.2491 |
| 031 | chimeras L6 | n2v-sort | list | 0.3301 |

Details:

| #XP | test set  | setup | train over | eval metric |
| --- | --- | --- | --- | --- |
| 013 | nonces | n2v-asi | list | 0.03698 |
| 014 | nonces | n2v-cwe | list | 0.03656 |
| 015 | nonces | n2v-cwe | set | 0.03655 |
| 016 | nonces | n2v-sort | list | 0.03668 |
| 017 | nonces | n2v-sort | set | 0.03643 |
| 018 | L2 | n2v-asi | list | 0.2786 |
| 019 | L2 | n2v-cwe | list | 0.2621 |
| 020 | L2 | n2v-cwe | set | 0.2648 |
| 021 | L2 | n2v-sort | list | 0.2712 |
| 022 | L2 | n2v-sort | set | 0.2720 |
| 023 | L4 | n2v-asi | list | 0.2436 |
| 024 | L4 | n2v-cwe | list | 0.2474 |
| 025 | L4 | n2v-cwe | set | 0.2455 |
| 026 | L4 | n2v-sort | list | 0.2491 |
| 027 | L4 | n2v-sort | set | 0.2445 |
| 028 | L6 | n2v-asi | list | 0.3255 |
| 029 | L6 | n2v-cwe | list | 0.3280 |
| 030 | L6 | n2v-cwe | set | 0.3216 |
| 031 | L6 | n2v-sort | list | 0.3331 |
| 032 | L6 | n2v-sort | set | 0.3285 |

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

The baseline we are trying to beat is:

| nonces | chimeras L2 | chimeras L4 | chimeras L6 |
| --- | --- | --- | --- |
| 0.03655 | 0.2759 | 0.2400 | 0.3312 |

Details:

| #XP | alpha | beta | kappa | train over| train filter | nonces | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 033 | 1 | 1000 | 1 | list | - | 0.03654 | 0.2781 | 0.2505 | 0.3246 |
| 034 | 1 | 1000 | 1 | list | cwe | 0.03678 | 0.2767 | 0.2461 | 0.3305 |
| 035 | 1 | 1000 | 1 | set | - | 0.03647 | 0.2786 | 0.2455 | 0.3225 |
| 036 | 1 | 1000 | 1 | set | cwe | 0.03674 | 0.2787 | 0.2457 | 0.3276 |
| 037 | 2 | 1000 | 1 | list | - | 0.03744 | 0.2480 | 0.1821 | 0.1690 |
| 038 | 3 | 1000 | 1 | list | - | 0.03808 | 0.2141 | 0.1152 | 0.1428 |
| 138 | 3 | 1000 | 2 | list | - | 0.03762 | 0.2520 | 0.1984 | 0.1797 |
| 238 | 3 | 1500 | 2 | list | - | 0.03732 | 0.2538 | 0.1798 | 0.2153 |
| 338 | 3 | 500 | 2 | list | - | 0.03613 | 0.2710 | 0.1859 | 0.1714 |
| 438 | 3 | 1000 | 3 | list | - | 0.03713 | 0.2791 | 0.2341 | 0.2751 |
| A38 | 3 | 750 | 3 | list | - | 0.03570 | 0.2810 | 0.2466 | 0.3157 |
| B38 | 3 | 1250 | 3 | list | - | 0.03689 | 0.2837 | 0.2355 | 0.2638 |
| C38 | 3 | 1100 | 3 | list | - | 0.03700 | 0.2800 | 0.2202 | 0.2613 |
| D38 | 3 | 1200 | 3 | list | - | 0.03688 | 0.2827 | 0.2300 | 0.2539 |
| E38 | 3 | 1300 | 3 | list | - | 0.03691 | 0.2837 | 0.2342 | 0.2623 |
| F38 | 3 | 900 | 3 | list | - | 0.03687 | 0.2795 | 0.2396 | 0.2884 |
| G38 | 3 | 800 | 3 | list | - | 0.03623 | 0.2784 | 0.2394 | 0.3190 |
| 538 | 3 | 1500 | 3 | list | - | 0.03692 | 0.2701 | 0.2138 | 0.2843 |
| 638 | 3 | 500 | 3 | list | - | 0.03557 | 0.2819 | 0.2455 | 0.3224 |
| 738 | 4 | 1000 | 1 | list | - | 0.03442 | 0.1508 | 0.0407 | 0.1339 |
| 838 | 4 | 500 | 3 | list | - | 0.03494 | 0.2801 | 0.2230 | 0.2945 |
| 039 | 5 | 1000 | 1 | list | - | 0.03423 | 0.1667 | 0.0934 | 0.1082 |
| 040 | 10 | 1000 | 1 | list | - | 0.0288 | 0.1010 | 0.0052 | 0.0149 |
| 041 | 0.5 | 1000 | 1 | list | - | 0.03647 | 0.2728 | 0.2424 | 0.3263 |
| 042 | 0.3 | 1000 | 1 | list | - | 0.03653 | 0.2728 | 0.2413 | 0.3248 |
| 043 | 0.1 | 1000 | 1 | list | - | 0.03655 | 0.2738 | 0.2400 | 0.3256 |
| 044 | 1 | 1000 | 2 | list | - | 0.03638 | 0.2779 | 0.2434 | 0.3266 |
| 045 | 1 | 1000 | 3 | list | - | 0.03643 | 0.2733 | 0.2457 | 0.3294 |
| 046 | 1 | 500 | 1 | list | - | 0.03649 | 0.2765 | 0.2484 | 0.3305 |
| 047 | 1 | 500 | 2 | list | - | 0.03647 | 0.2730 | 0.2434 | 0.3253 |
| 048 | 1 | 500 | 3 | list | - | 0.03648 | 0.2743 | 0.2418 | 0.3277 |
| 049 | 1 | 1500 | 1 | list | - | 0.03652 | 0.2747 | 0.2524 | 0.3204 |
| 050 | 1 | 1500 | 2 | list | - | 0.03638 | 0.2759 | 0.2455 | 0.3209 |
| 051 | 1 | 1500 | 3 | list | - | 0.03637 | 0.2718 | 0.2442 | 0.3258 |

### Quality of the background model
The performance of the cwe-based approach (filter and learning rate) are
conditioned on the 'quality' of the background model probability distributions.
Here we question whether hyperparameters of the background model
(window size, subsampling rate, min count)
positively impact cwe-based learning.

| #XP | window | sample | min  | nonces | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 033 | 5 | 10-3 | 50 | 0.03654 | 0.2781 | 0.2505 | 0.3246 |
| 052 | 5 | 10-5 | 50 | 0.03810 | 0.2481 | 0.2270 | 0.2543 |
| 053 | 10 | 10-3 | 50 | 0.03489 | 0.2493 | 0.2560 | 0.3343 |
| 054 | 15 | 10-3 | 50 | 0.03465 | 0.2650 | 0.2415 | 0.3193 |
| 055 | 15 | 10-2 | 50 | 0.03224 | 0.2822 | 0.2567 | 0.3138 |
| 056 | 15 | 10-3 | 5 | 0.03260 | 0.2343 |  | |
| 057 | 15 | 10-3 | 10 | 0.03461 | 0.2462 | 0.2101 | |
| 058 | 15 | 10-3 | 15 | 0.03497 | 0.2468 | 0.2332 | 0.2923 |
| 059 | 15 | 10-3 | 20 | 0.03374 | 0.2328 | 0.2586 | 0.3165 |
| 060 | 15 | 10-3 | 25 | 0.03536 | 0.2219 | 0.2305 | 0.3091 |
| 061 | 15 | 10-3 | 100 | 0.03717 | 0.2561 | 0.2375 | 0.3070 |
| 062 | 15 | 10-5 | 50 | 0.03768 | 0.2687 | 0.2299 | 0.2843 |
| 063 | 15 | 10-7 | 50 | 0.02997 | 0.2714 | 0.2244 | 0.2633 |
| 066 | 20 | 10-3 | 50 | 0.03350 | 0.2683 | 0.2489 | 0.2949 |
| 067 | 50 | 10-3 | 50 | 0.03250 | 0.2480 | 0.2164 | 0.2915 |


### Robustness of our results
Testing all our results (CWI SUM baseline + best CWI_ALPHA model) on H&B's
background model:

| #XP | System | Nonces  | Chimeras L2 | Chimeras L4 | Chimeras L6 |
| --- | --- | --- | --- | --- | --- |
| 068 | Sum over list no filter | 0.0095 | 0.3066 | 0.3337 | 0.3260 |
| 069 | Sum over set cwi filter | 0.04160 | 0.3271 | 0.3538 | 0.3890 |
| 070 | n2v asi | 0.05392 | 0.3092 | 0.3268 | 0.3871 |
| 071 | n2v cwi alpha (a = 1 b = 1000 k = 1) | 0.04729 | 0.3423 | 0.3369 | 0.3765 |
| 072 | n2v cwi alpha (a = 3 b = k = 3) | | | | |
| 073 | n2v cwi alpha with  | | | | |


### Going further: is context entropy useful in itself?
Two questions I would like to ask if I still have time:
1. Can context entropy help us determine 'minimally-informative' sentences?
2. Does context entropy correlates well with our overall results?

### Notes
try out with very low learning rate
if sum is doing everything should see a decrease in perf if increasing learning rate

MRR versus density of gold vectors. Claculate distance to nearest neighbors.
