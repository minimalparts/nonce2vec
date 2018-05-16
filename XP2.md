# Experiments (2nd round)

Refactoring:
- window size = full sentence
- no window decay and sample decay (useless on nonces dataset anyway)
- fixed bug with alpha
- summing on whole context (not set of context)
- 'duran' nonce is tested twice in test set. We removed the second occurrence

## Nonces

### Replication with Aurélie's background model
| XP | setup | MRR |
| --- | --- | --- |
| 022 | sum-only + random filter on sum | |
| 023 | sum-only + random filter on both | |

### Filtering context
Filters:
1. No filter
2. Random filter
3. Self information filter
4. Context word entropy filter

| #XP | config | no filter | random | self > 22 | cwe > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 024 | Sum |  |  |  |  |
| 025 | N2V + SF |  |  |  |  |
| 026 | N2V + SF + TF |  |  |  |  |

SF: sum filter
TF: train filter

### Impact of window size of info model (sum-only, sample 1e-3, min 50)

| #XP | window size | MRR |
| --- | --- | --- |
| | 5 | 0.03284 |
| | 10 | 0.03215 |
| | 15 | 0.03147 |
| | 20 | 0.03045 |
| | 50 | 0.03139 |

### Impact of min_count of info model (sum-only, window 15, sample 1e-3)

| #XP | min count | MRR |
| --- | --- | --- |
| | 5 | 0.02839* |
| | 10 | 0.03081* |
| | 25 | 0.03211 |
| | 50 | 0.03147* |
| | 100 | 0.03331* |

### Impact of sampling rate of info model (sum-only, with window 15, min 50)

| XP | sample | MRR |
| --- | --- | --- |
| | 1e-2 | 0.03055* |
| | 1e-3 | 0.03147* |
| | 1e-5 | 0.03285* |
| | 1e-7 | 0.02968* |
| | 1e-9 | 0.00614* |

### Impact of desc. sorting with exp_decay (no filtering on train)

- win5, sample 1e-3: 0.03317
- win20, sample 1e-3: 0.03117
- win15, sample 1e-3: 0.03207
- win15, sample 1e-2: 0.02936
- win15, sample 1e-5: 0.03333
- win10, sample 1e-5: 0.03038
- win5, sample 1e-5: *

### Impact of filtering and sorting on train with exp_decay

On standard model (win 5, min 50, sample 1e-3)
- with sorting no filter: 0.02848
- with sorting and filter: 0.03263
- no sorting no filter: 0.03375*
- no sorting filter: 0.03255*

## Lambda
Sorting desc no train filter:
- XP19 lambda 70:
- XP20 lambda 100:
- XP21 lambda 150:
- XP22 lambda 500:


### Impact of learning rate on training
Standard model (win 5, sample 1e-3, min 50, lambda 70) sum filter cwe
no train filter, sorting desc

| 0.5 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 0.02848* | 0.02251* | 0.01736* | 0.01747* | 0.01602* | 0.01490* | 0.01627* |

### Impact of alpha decay function

XP With kappa = 1, alpha = 1, beta = 1000 (no train filter): 0.03281
XP With kappa = 1, alpha = 1, beta = 1000 + train filter: 0.03294

## Chimeras

### Filtering

Sum-only:

| #XP | L | no filter | random | self > 22 | cwe > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 001 | L2 | 0.1832 | 0.2115 | 0.2157 | 0.2557 |
| 002 | L4 | 0.1367 | 0.1722 | 0.1780 | 0.2383 |
| 003 | L6 | 0.1290 | 0.1616 | 0.1999 | 0.3312 |

Train with sum filter (no train filter)

| #XP | L | no filter | random | self > 22 | cwe > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 004 | L2 | 0.0223 | 0.1121 | 0.0793 | 0.0609 |
| 005 | L4 | 0.1201 | 0.1330 | 0.0638 | 0.1078 |
| 006 | L6 | 0.1760 | 0.1538 | 0.1586 | 0.2366 |

Train with sum filter + train filter

| #XP | L | no filter | random | self > 22 | cwe > 0 |
| --- | --- | --- | --- | --- | --- | --- |
| 013 | L2 |  | 0.2003 | 0.2196 | 0.2486 |
| 014 | L4 |  | 0.1232 | 0.1673 | 0.2476 |
| 015 | L6 |  | 0.0598 | 0.1562 | 0.3292 |

### Sorting

Sorting is desc order, no train filter, sum filter, standard config

| #XP | L | RHO |
| --- | --- | --- |
| 007 | L2 | 0.0664 |
| 008 | L4 | 0.1551 |
| 009 | L6 |  |

Sorting is desc order with train filter + sum filter in standard config

| #XP | L | RHO |
| --- | --- | --- |
| 016 | L2 | 0.2524 |
| 017 | L4 | 0.2420 |
| 018 | L6 |  |

### alpha decay

With kappa = 1, alpha = 1, beta = 1000, sum filter, no train filter

| #XP | L | RHO |
| --- | --- | --- |
| 010 | L2 | 0.2578 |
| 011 | L4 | 0.2410 |
| 012 | L6 |  |

With kappa = 1, alpha = 1, beta = 1000, sum filter + train filter

| #XP | L | RHO |
| --- | --- | --- |
| 027 | L2 | 0.2622 |
| 028 | L4 | 0.2389 |
| 029 | L6 |  |
