# XP summary

## Original results

### Definitional

| System | Paper | Median Rank | MRR |
| --- | --- |
| Nonce2Vec | Herbelot and Baroni (2017) | 623 | 0.04907 |
| A La Carte | Khodak et al. (2018) | 165.5 | 0.07058 |
| Form-Context | Schick and Schütze (2019) | 49 | 0.17537 |

### Chimeras
| System | L | RHO |
| --- | --- | --- |
| N2V | L2 | 0.3320 |
| ALC | L2 | 0.3634 |
| N2V | L4 | 0.3668 |
| ALC | L4 | 0.3844 |
| SUM | L6 | 0.4080 |
| N2V | L6 | 0.3890 |
| ALC | L6 | 0.3941 |

## Replication

MEN = 0.74

### Definitional

XP001 -- with reload (original)
MRR = 0.0477
Median Rank = 955

XP101 -- no reload
MRR = 0.0032
Median Rank = 78504

| #XP | Median Rank | MRR |
| --- | --- | --- |
| 001 | 955 | 0.0477 |
| 101 | 78504 | 0.0032 |

### Chimeras

| #XP | L | RHO |
| --- | --- | --- |
| 002 | L2 | 0.3409 |
| 003 | L4 | 0.3471 |
| 004 | L6 | 0.3300 |

## Informativeness

### Context filter

| #XP | Filter | Median Rank | MRR |
| --- | --- | --- | --- |
| 005 | no filter | 5969 | 0.0087 |
| 105 | no filter | 6110 | 0.0018 |
| 006 | random | 3047 | 0.0221 |
| 106 | random | 3212 | 0.0065 |
| 007 | self | 1769 | 0.0242 |
| 107 | self | 1923 | 0.0120 |
| 008 | cwi | 935 | 0.0374 |
| 108 | cwi | 935 | 0.0321 |

| #XP | Filter | L | RHO |
| --- | --- | --- |
| 009 | no filter | L2 | 0.3075 |
| 010 | random | L2 | 0.3411 |
| 011 | self | L2 | 0.3442 |
| 012 | cwi | L2 | 0.3016 |
| 013 | no filter | L4 | 0.3208 |
| 014 | random | L4 | 0.3651 |
| 015 | self | L4 | 0.3635 |
| 016 | cwi | L4 | 0.3837 |
| 017 | no filter | L6 | 0.2933 |
| 018 | random | L6 | 0.3542 |
| 019 | self | L6 | 0.3541 |
| 020 | cwi | L6 | 0.4128 |

### Adaptative learning rate

| #XP | Filter | Median Rank | MRR |
| --- | --- | --- | --- |
| 008 | cwi sum | 935 | 0.0374 |
| 108 | cwi sum | 935 | 0.0321 |
| 001 | n2v as-is | 955 | 0.0477 |
| 101 | n2v as-is | 78504 | 0.0032 |
| 021 | n2v cwi init | 540 | 0.0493 |
| 121 | n2v cwi init | 94889 | 0.0064 |
| 022 | n2v cwi alpha | 763 | 0.0404 |
| 122 | n2v cwi alpha | 945 | 0.0336 |

| #XP | Model | L | RHO |
| --- | --- | --- |
| 011 | self sum | L2 | 0.3442 |
| 002 | n2v as-is | L2 | 0.3409 |
| 023 | n2v cwi init | L2 | 0.3009 |
| 024 | n2v cwi alpha | L2 | 0.3084 |
| 016 | cwi sum | L4 | 0.3837 |
| 003 | n2v as-is | L4 | 0.3471 |
| 025 | n2v cwi init | L4 | 0.3432 |
| 026 | n2v cwi alpha | L4 | 0.3919 |
| 020 | cwi sum | L6 | 0.4128 |
| 004 | n2v as-is | L6 | 0.3300 |
| 027 | n2v cwi init | L6 | 0.3585 |
| 028 | n2v cwi alpha | L6 | 0.4077 |

### Incremental learning on the chimera dataset
XP009 original RHO = 0.3075
RHO = 0.3155

### Shuffling test set in incremental learning

### Sampling alpha and beta parameters
