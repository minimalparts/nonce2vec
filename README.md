# Nonce2Vec

## Replication
| #XP | Background| Nonces | L2 | L4 | L6 |
| --- | --- | --- | --- | --- | --- |
| 001 | Ours | 0.02954 | 0.2049 | 0.1895 | 0.1844 |
| 002 | H&B | 0.04303 | 0.3336 | 0.3438 | 0.3922 |


Sum over set all the time

| #XP | Filter | Ours | H&B |
| --- | --- | --- | --- |
| 003 | none | 0.02031 | 0.01791 |
| 004 | random | 0.02743 | 0.02610 |
| 005 | self | 0.03077 | 0.03393 |
| 006 | cwi |  |  |


| #XP | System | Ours | H&B |
| --- | --- | --- | --- |
| 006 | cwi sum |  |  |
| 007 | n2v as-is | 0.02954 | 0.04303 |
| 008 | n2v cwi init |  |  |
| 009 | n2v cwi alpha |  |  |


On our background model

| #XP | System | L2 | L4 | L6 |
| --- | --- | --- | --- | --- |
| 010 | self sum | 0.2213 | 0.1785 | 0.2113 |
| 011 | cwi sum |  |  |
| 012 | n2v as-is | 0.2049 | 0.1895 | 0.1844 |
| 013 | n2v cwi init |  |  |
| 014 | n2v cwi alpha |  |  |

On H&B's

| #XP | System | L2 | L4 | L6 |
| --- | --- | --- | --- | --- |
| 015 | self sum | 0.3352 | 0.3485 | 0.4095 |
| 016 | cwi sum |  |  |
| 017 | n2v as-is | 0.3336 | 0.3438 | 0.3922 |
| 018 | n2v cwi init |  |  |
| 019 | n2v cwi alpha |  |  |
