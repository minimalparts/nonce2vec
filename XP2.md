# Experiments (2nd round)

Refactoring:
- window size = full sentence
- no window decay and sample decay (useless on nonces dataset anyway)
- fixed bug with alpha
- summing on whole context (not set of context)
- 'duran' nonce is tested twice in test set. We removed the second occurrence

## Replication with Aurélie's background model
with random subsampling (sum + train)
SUM: 0.00909 (no-reload) 0.02074657 (reload)
N2V: 0.00442 (no-reload)

with self filter:
SUM: 0.0276477 (with reload) 0.0141539 (no reload)
N2V: 0.04198 (with reload) 0.015453 (no reload)

no filter:
SUM: 0.00325612 (no-reload)
N2V: 0.004155 (no-reload)

CWE filter on both sum and train:
SUM: 0.0299246 (no reload)
N2V: 0.02744 (no-reload)

## Filtering context
Filters:
1. No filter
2. Random filter
3. Self information filter
4. Context word entropy filter

| config | no filter | random | self 22 | CWE > 0 (win5) |
| --- | --- | --- | --- | --- | --- |
| sum-only | 0.01955 | 0.02600 | 0.02762 | 0.03284 |
| sum filter + n2v train no filter + no sort | 0.01985* | 0.02758* | 0.02901* | 0.03375* |
| sum filter + n2v filter (same) + no sort | 0.01985* | 0.02787* | 0.02908* | 0.03255* |

## Impact of window size of info model (sum-only, sample 1e-3, min 50)

| window size | MRR |
| --- | --- |
| 5 | 0.03284 |
| 10 | 0.03215 |
| 15 | 0.03147 |
| 20 | 0.03045 |
| 50 | 0.03139 |

## Impact of min_count of info model (sum-only, window 15, sample 1e-3)

| min count | MRR |
| --- | --- |
| 5 | 0.02839 |
| 10 | 0.03081 |
| 25 | 0.03211 |
| 50 | 0.03147 |
| 100 | 0.03331 |

## Impact of sampling rate of info model (sum-only, with window 15, min 50)

| sample | MRR |
| --- | --- |
| 1e-2 | * |
| 1e-3 | 0.03147 |
| 1e-5 | 0.03285 |
| 1e-7 | 0.02968 |
| 1e-9 | 0.00614 |

## Impact of desc. sorting with exp_decay (no filtering on train)

- win5, sample 1e-3: 0.03317
- win20, sample 1e-3: 0.03117
- win15, sample 1e-3: 0.03207
- win15, sample 1e-2:
- win15, sample 1e-5: 0.03333
- win10, sample 1e-5:
- win5, sample 1e-5:

## Impact of filtering and sorting on train with exp_decay

On standard model (win 5, min 50, sample 1e-3)
- with sorting no filter: 0.03317*
- with sorting and filter: 0.03261*
- no sorting no filter: 0.03375*
- no sorting filter: 0.03255*


## Impact of learning rate on training
Standard model (win 5, sample 1e-3, min 50, lambda 70) sum filter cwe
no train filter, sorting desc

| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|  | | | | | |

## Impact of alpha decay function
