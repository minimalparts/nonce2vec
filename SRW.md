# XP summary

## Original results

### Definitional

| System | Paper | Median Rank | MRR |
| --- | --- |
| Nonce2Vec | Herbelot and Baroni (2017) | 623 | 0.04907 |
| A La Carte| Khodak et al. (2018) | 165.5 | 0.07058 |
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

MEN = .74

### Definitional

XP001 -- with reload (original)
MRR = 0.048
Median Rank = 955

XP101 -- no reload
MRR = 0.003
Median Rank = 78504

| #XP | Median Rank | MRR |
| --- | --- | --- |
| 001 | 955 | 0.048 |
| 101 | 78504 | 0.003 |

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
| 008 | cwi | 1414 | 0.0351 |
| 108 | cwi | 1430 | 0.0241 |

| #XP | Filter | L | RHO |
| --- | --- | --- |
| 009 | no filter | L2 |  |
| 010 | random | L2 |  |
| 011 | self | L2 |  |
| 012 | cwi | L2 |  |
| 013 | no filter | L4 |  |
| 014 | random | L4 |  |
| 015 | self | L4 |  |
| 016 | cwi | L4 |  |
| 017 | no filter | L6 |  |
| 018 | random | L6 |  |
| 019 | self | L6 |  |
| 020 | cwi | L6 |  |

### Adaptative learning rate

| #XP | Filter | Median Rank | MRR |
| --- | --- | --- | --- |
| 008 | cwi sum |  | |
| 108 | cwi sum | 1430 | 0.0241 |
| 001 | n2v as-is | 955 | 0.048 |
| 101 | n2v as-is | 78504 | 0.003 |
| 021 | n2v cwi init |  | |
| 121 | n2v cwi init |  | |
| 022 | n2v cwi alpha |  | |
| 122 | n2v cwi alpha |  | |

| #XP | Model | L | RHO |
| --- | --- | --- |
| 012 | cwi sum | L2 |  |
| 002 | n2v as-is | L2 |  |
| 023 | n2v cwi init | L2 |  |
| 024 | n2v cwi alpha | L2 |  |
| 016 | cwi sum | L4 |  |
| 003 | n2v as-is | L4 |  |
| 025 | n2v cwi init | L4 |  |
| 026 | n2v cwi alpha | L4 |  |
| 020 | cwi sum | L6 |  |
| 004 | n2v as-is | L6 |  |
| 027 | n2v cwi init | L6 |  |
| 028 | n2v cwi alpha | L6 |  |
