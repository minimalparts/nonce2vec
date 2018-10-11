# Experiments

## Background

### With the gensim implementation

On the 'old' wikidump of Herbelot and Baroni (2017)

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



## Definitional

### Replication

| XP | Background | MRR |
| --- | --- | --- |
| XXX | HB (orig. 0.13) | 0.04907 |
| 009 | HB (3.4) | |
| 010 | skip.repl.old.wikidump | |
| 011 | skip.repl.new.wikidump | |
| 012 | skip.best.new.wikidump | |
| 013 | cbow.best.new.wikidump | |

### Filtering in sum-only mode

Using the best skipgram background model

| XP  | filter | MRR |
| --- | --- | --- |
| 014 | no-filter | |
| 015 | random | |
| 016 | self | |
| 017 | cwi | |

Using the best cbow background model

| XP  | filter | MRR |
| --- | --- | --- |
| 018 | no-filter | |
| 019 | random | |
| 020 | self | |
| 021 | cwi | |

### Training

original results (as per EMNLP submission)

| XP  | mode | MRR |
| --- | --- | --- |
| 022 | cwi sum | |
| 023 | n2v as-is | |
| 024 | n2v cwi init | |
| 025 | n2v cwi alpha | |

replication with skipgram on same hyperparameters trained on new wikidump
(with cbow for informativeness trained on same hyperparameters as skipgram)

| XP  | mode | MRR |
| --- | --- | --- |
| 026 | cwi sum | |
| 027 | n2v as-is | |
| 028 | n2v cwi init | |
| 029 | n2v cwi alpha | |

combining best skipgram with best cbow on  new wikidump

| XP  | mode | MRR |
| --- | --- | --- |
| 030 | cwi sum | |
| 031 | n2v as-is | |
| 032 | n2v cwi init | |
| 033 | n2v cwi alpha | |

best cbow all-the-way (for both background and informativeness)

| XP  | mode | MRR |
| --- | --- | --- |
| 034 | cwi sum | |
| 035 | n2v as-is | |
| 036 | n2v cwi init | |
| 037 | n2v cwi alpha | |

## Chimeras

### Replication

On L2

| XP | Background | RHO |
| --- | --- | --- |
| XXX | HB (orig. 0.13) |  |
| 038 | HB (3.4) | |
| 039 | skip.repl.old.wikidump | |
| 040 | skip.repl.new.wikidump | |
| 041 | skip.best.new.wikidump | |
| 042 | cbow.best.new.wikidump | |

On L4

| XP | Background | RHO |
| --- | --- | --- |
| XXX | HB (orig. 0.13) |  |
| 043 | HB (3.4) | |
| 044 | skip.repl.old.wikidump | |
| 045 | skip.repl.new.wikidump | |
| 046 | skip.best.new.wikidump | |
| 047 | cbow.best.new.wikidump | |

On L6

| XP | Background | RHO |
| --- | --- | --- |
| XXX | HB (orig. 0.13) |  |
| 048 | HB (3.4) | |
| 049 | skip.repl.old.wikidump | |
| 050 | skip.repl.new.wikidump | |
| 051 | skip.best.new.wikidump | |
| 052 | cbow.best.new.wikidump | |

### Filtering

On L2

| XP  | filter | RHO |
| --- | --- | --- |
| 0 | no-filter | |
| 0 | random | |
| 0 | self | |
| 0 | cwi | |

On L4

| XP  | filter | RHO |
| --- | --- | --- |
| 0 | no-filter | |
| 0 | random | |
| 0 | self | |
| 0 | cwi | |

On L6

| XP  | filter | RHO |
| --- | --- | --- |
| 0 | no-filter | |
| 0 | random | |
| 0 | self | |
| 0 | cwi | |

### Training

On L2

On L4

On L6
