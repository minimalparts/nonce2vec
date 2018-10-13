# Experiments

## Background

### With the gensim implementation

On the 'old' wikidump of Herbelot and Baroni (2017?)

| XP | mode | alpha | neg | window | sample | epochs | min count | size | MEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 001 | skip | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 400 | 0.75 |
| 002 | skip | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 700 | 0.76 |
| 003 | cbow | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 400 | 0.70 |
| 004 | cbow | 0.010 | 10 | 50 | 1e-3 | 5 | 50 | 1000 | 0.75 |

On the 'new' wikidump (2018.09.20)

| XP | mode | alpha | neg | window | sample | epochs | min count | size | MEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 005 | skip | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 400 | 0.75 |
| 006 | skip | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 700 | 0.75 | (slightly better than above by 0.005 points)
| 007 | cbow | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 400 | 0.69 |
| 008 | cbow | 0.010 | 10 | 50 | 1e-3 | 5 | 50 | 1000 | 0.74 |

### With the Tensorflow implementation

On the 'new' wikidump (2018.09.20)

| XP | mode | alpha | neg | window | sample | epochs | min count | size | MEN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 005 | skip | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 400 |  |
| 006 | skip | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 700 |  |
| 007 | cbow | 0.025 | 5 | 5 | 1e-3 | 5 | 50 | 400 |  |
| 008 | cbow | 0.010 | 10 | 50 | 1e-3 | 5 | 50 | 1000 |  |

## Definitional

### Replication

| XP | Background | MRR |
| --- | --- | --- |
| XXX | HB (orig. 0.13) | 0.04907 |
| 009 | HB (3.4) | 0.04303 |
| 010 | skip.repl.old.wikidump | 0.04417 |
| 011 | skip.repl.new.wikidump | 0.04400 |

### Filtering in sum-only mode

original results (as per EMNLP submission)

| filter | RP | HB |
| --- | --- | --- |
| no-filter | 0.02031 | 0.01791 |
| random | 0.02743 | 0.02610 |
| self | 0.03077 | 0.03393 |
| cwi | 0.03741 | 0.04054 |

replication (standard skipgram model on old wikidump)

| XP  | filter | MRR |
| --- | --- | --- |
| 014 | no-filter | 0.01151 |
| 015 | random | 0.02187 |
| 016 | self | 0.02952 |
| 017 | cwi | 0.03352 | info model is cbow with same hyperparams as skipgram
| 117 | cwi | 0.03780 | info model is best cbow

Using the standard skipgram background model on latest wikidump

| XP  | filter | MRR |
| --- | --- | --- |
| 018 | no-filter | 0.01085 |
| 019 | random | 0.02146 |
| 020 | self | 0.02658 |
| 021 | cwi | 0.03581 | info model is cbow with same hyperparams as skipgram
| 121 | cwi | 0.03788 | info model is best cbow

Using the best cbow background model (on HOLD)

### Training

original results (as per EMNLP submission)

| XP  | mode | MRR |
| --- | --- | --- |
| XXX | cwi sum | |
| XXX | n2v as-is | |
| XXX | n2v cwi init | |
| XXX | n2v cwi alpha | |

replication on old wikidump

| XP  | mode | MRR |
| --- | --- | --- |
| 117 | cwi sum | 0.03780 |
| 023 | n2v as-is | 0.04417 |
| 024 | n2v cwi init | 0.04888 | with same cbow
| 124 | n2v cwi init | 0.04369 | with best cbow
| 025 | n2v cwi alpha | 0.04049 | same cbow
| 125 | n2v cwi alpha | 0.03836 | best cbow

replication with skipgram on same hyperparameters trained on new wikidump

| XP  | mode | MRR |
| --- | --- | --- |
| 121 | cwi sum | 0.03788 |
| 027 | n2v as-is | 0.04400 |
| 028 | n2v cwi init | 0.04526 |
| 128 | n2v cwi init | 0.04901 |
| 029 | n2v cwi alpha | 0.03992 |
| 129 | n2v cwi alpha | 0.03862 |

## Chimeras

### Replication

On L2

| XP | Background | RHO |
| --- | --- | --- |
| XXX | HB (orig. 0.13) | 0.3320 |
| 038 | HB (3.4) | 0.3336 |
| 039 | skip.repl.old.wikidump | 0.3036 |
| 040 | skip.repl.new.wikidump | 0.2982 |

On L4

| XP | Background | RHO |
| --- | --- | --- |
| XXX | HB (orig. 0.13) | 0.3668 |
| 043 | HB (3.4) | 0.3438 |
| 044 | skip.repl.old.wikidump | 0.3518 |
| 045 | skip.repl.new.wikidump | 0.3675 |

On L6

| XP | Background | RHO |
| --- | --- | --- |
| XXX | HB (orig. 0.13) | 0.3890 |
| 048 | HB (3.4) | 0.3922 |
| 049 | skip.repl.old.wikidump | 0.3560 |
| 050 | skip.repl.new.wikidump | 0.3660 |

### Filtering

On L2

| XP  | filter | RHO |
| --- | --- | --- |
| 051 | no-filter | 0.3061 |
| 052 | random | 0.3400 |
| 053 | self | 0.3465 |
| 054 | cwi | 0.3162 |

On L4

| XP  | filter | RHO |
| --- | --- | --- |
| 055 | no-filter | 0.3581 |
| 056 | random | 0.3649 |
| 057 | self | 0.3623 |
| 058 | cwi | 0.3598 |

On L6

| XP  | filter | RHO |
| --- | --- | --- |
| 059 | no-filter | 0.3445 |
| 060 | random | 0.3549 |
| 061 | self | 0.3637 |
| 062 | cwi | 0.3981 |

### Training

On L2
| XP  | mode | MRR |
| --- | --- | --- |
| 040 | n2v as-is | 0.2982 |
| 063 | n2v cwi init |  | with same cbow
| 064 | n2v cwi init |  | with best cbow
| 065 | n2v cwi alpha |  | same cbow
| 066 | n2v cwi alpha |  | best cbow

On L4
| XP  | mode | MRR |
| --- | --- | --- |
| 045 | n2v as-is | 0.3675 |
| 067 | n2v cwi init |  | with same cbow
| 068 | n2v cwi init |  | with best cbow
| 069 | n2v cwi alpha |  | same cbow
| 070 | n2v cwi alpha |  | best cbow

On L6
| XP  | mode | MRR |
| --- | --- | --- |
| 050 | n2v as-is | 0.3660 |
| 071 | n2v cwi init |  | with same cbow
| 072 | n2v cwi init |  | with best cbow
| 073 | n2v cwi alpha |  | same cbow
| 074 | n2v cwi alpha |  | best cbow
