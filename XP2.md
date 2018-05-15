# Experiments (2nd round)

Refctoring:
- window size = full sentence
- no window decay and sample decay (useless on nonces dataset anyway)
- fixed bug with alpha
- summing on whole context (not set of context)
- 'duran' nonce is tested twice in test set. We removed the second occurence

## Replication with Aurélie's background model
with random subsampling (sum + train)
SUM: 0.00909 (no-reload) 0.02074657 (reload)
N2V: 0.00442 (no-reload)

with self filter:
SUM: 0.0276477 (with reload) 0.0141539 (no reload)
N2V:

no filter:
SUM: 0.00325612 (no-reload)
N2V:

## Filtering context
Filters:
1. No filter
2. Random filter
3. Self information filter
4. Context word entropy filter

| config | no filter | random | self 22 | CWE > 0 (win5) |
| --- | --- | --- | --- | --- | --- |
| sum-only |  |  |  |  | |
| sum filter + n2v train no filter |
| sum filter + n2v filter (same) |

## Impact of window size of info model
| window size | MRR |
| --- | --- |
| 5 |  |
| 10 |  |
| 15 |  |
| 20 |  |
| 50 |  |

## Impact of min_count of info model
| min count | MRR |
| --- | --- |
| 5 |  |
| 10 |  |
| 25 |  |
| 50 |  |
| 100 |  |

## Impact of sampling rate of info model

| sample | MRR |
| --- | --- |
| 1e5 |  |
| 1e3 |  |
| 1 |  |
| 1e-3 |  |
| 1e-5 |  |
| 1e-10 |  |


## Impact of learning rate on training

## Impact of alpha decay function
