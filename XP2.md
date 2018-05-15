# Experiments (2nd round)

Refctoring:
- window size = full sentence
- no window decay and sample decay (useless on nonces dataset anyway)
- fixed bug with alpha
- summing on whole context (not set of context)

## Replication with Aurélie's background model
with random subsampling (sum + train)
SUM:
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

## Impact of sampling rate of info model

| sample | MRR |
| --- | --- |
| 10000 |  |
| 50000 |  |
| 100000 |  |
| 500000 |  |
| 1000000 |  |

## Impact of learning rate on training

## Impact of alpha decay function 
