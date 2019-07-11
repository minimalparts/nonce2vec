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

| #XP | Median Rank | MRR |
| --- | --- | --- |
| 001 | 955 | 0.0477 |
| 101 | 78504 | 0.0031 |

### Chimeras

| #XP | L | RHO |
| --- | --- | --- |
| 002 | L2 | 0.3412 |
| 102 | L2 | 0.1431 |
| 003 | L4 | 0.3514 |
| 103 | L4 | 0.1045 |
| 004 | L6 | 0.4077 |
| 104 | L6 | 0.1450 |

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
| 009 | no filter | L2 | 0.3047 |
| 109 | no filter | L2 | 0.3047 |
| 010 | random | L2 | 0.3358 |
| 110 | random | L2 | 0.3358 |
| 011 | self | L2 | 0.3455 |
| 111 | self | L2 | 0.3455 |
| 012 | cwi | L2 | 0.3074 |
| 112 | cwi | L2 | 0.3074 |
| 013 | no filter | L4 | 0.3288 |
| 113 | no filter | L4 | 0.3288 |
| 014 | random | L4 | 0.3717 |
| 114 | random | L4 | 0.3717 |
| 015 | self | L4 | 0.3638 |
| 115 | self | L4 | 0.3638 |
| 016 | cwi | L4 | 0.3739 |
| 116 | cwi | L4 | 0.3739 |
| 017 | no filter | L6 | 0.3063 |
| 117 | no filter | L6 | 0.3063 |
| 018 | random | L6 | 0.3584 |
| 118 | random | L6 | 0.3584 |
| 019 | self | L6 | 0.3651 |
| 119 | self | L6 | 0.3651 |
| 020 | cwi | L6 | 0.4243 |
| 120 | cwi | L6 | 0.4243 |

### Adaptative learning rate

| #XP | Filter | Median Rank | MRR |
| --- | --- | --- | --- |
| 008 | cwi sum |  |  |
| 108 | cwi sum |  |  |
| 001 | n2v as-is |  |  |
| 101 | n2v as-is |  |  |
| 021 | n2v cwi init | 540 | 0.0493 |
| 121 | n2v cwi init | 90056 | 0.0064 |
| 022 | n2v cwi alpha | 763 | 0.0404 |
| 122 | n2v cwi alpha | 944 | 0.0336 |

| #XP | Model | L | RHO |
| --- | --- | --- |
| 011 | self sum | L2 |  |
| 111 | self sum | L2 |  |
| 002 | n2v as-is | L2 |  |
| 102 | n2v as-is | L2 |  |
| 023 | n2v cwi init | L2 | 0.3002 |
| 123 | n2v cwi init | L2 | 0.1501 |
| 024 | n2v cwi alpha | L2 | 0.3129 |
| 124 | n2v cwi alpha | L2 | 0.2574 |
| 016 | cwi sum | L4 |  |
| 116 | cwi sum | L4 |  |
| 003 | n2v as-is | L4 |  |
| 103 | n2v as-is | L4 |  |
| 025 | n2v cwi init | L4 | 0.3482 |
| 125 | n2v cwi init | L4 | 0.1347 |
| 026 | n2v cwi alpha | L4 | 0.3928 |
| 126 | n2v cwi alpha | L4 | 0.2741 |
| 020 | cwi sum | L6 |  |
| 120 | cwi sum | L6 |  |
| 004 | n2v as-is | L6 |  |
| 104 | n2v as-is | L6 |  |
| 027 | n2v cwi init | L6 | 0.4218 |
| 127 | n2v cwi init | L6 | 0.1361 |
| 028 | n2v cwi alpha | L6 | 0.4181 |
| 128 | n2v cwi alpha | L6 | 0.1702 |

## Sampling alpha and beta parameters

### alpha (with beta = 1000)

In one-shot mode:

| model  | alpha | MR | MRR | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- |
| n2v cwi alpha | 1.0 |  |  |  |  |  |
| n2v cwi alpha | 0.1 |  |  |  |  |  |
| n2v cwi alpha | 10 |  |  |  |  |  |

In incremental mode:

| model  | alpha | MR | MRR | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- |
| n2v cwi alpha | 1.0 |  |  |  |  |  |
| n2v cwi alpha | 0.1 |  |  |  |  |  |
| n2v cwi alpha | 10 |  |  |  |  |  |

### beta (with alpha = 1.0)

In one-shot mode:

| model | beta | MR | MRR | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- |
| n2v cwi alpha | 1000 |  |  |  |  |  |
| n2v cwi alpha | 10000 |  |  |  |  |  |
| n2v cwi alpha | 100000 |  |  |  |  |  |

In incremental mode:

| model  | alpha | MR | MRR | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- |
| n2v cwi alpha | 1000 |  |  |  |  |  |
| n2v cwi alpha | 10000 |  |  |  |  |  |
| n2v cwi alpha | 100000 |  |  |  |  |  |

### alpha & beta
