# CMD

Training w2v background model on Wikipedia dump:
```
/home/kabbach/venv/bin/n2v train \
  --data /home/kabbach/witokit/data/wiki/enwiki.20190120.txt \
  --outputdir /home/kabbach/nonce2vec/models/ \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads 27 \
  --train-mode skipgram
```

Training informativeness background model with same hyperparameters as background model:
```
/home/kabbach/venv/bin/n2v train \
  --data /home/kabbach/witokit/data/wiki/enwiki.20190120.txt \
  --outputdir /home/kabbach/nonce2vec/models/ \
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads 27 \
  --train-mode cbow
```

XP001 -- with reload (original)
```
/home/kabbach/venv/bin/n2v test
  --on definitions \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication \
  --reload
```

XP101 -- no reload
```
/home/kabbach/venv/bin/n2v test
  --on definitions \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```

XP002
```
/home/kabbach/venv/bin/n2v test
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication \
  --reload
```

XP102:
```
/home/kabbach/venv/bin/n2v test
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```

XP003
```
/home/kabbach/venv/bin/n2v test
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication \
  --reload
```

XP103:
```
/home/kabbach/venv/bin/n2v test
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```

XP004
```
/home/kabbach/venv/bin/n2v test
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication \
  --reload
```

XP104:
```
/home/kabbach/venv/bin/n2v test
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/ \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt \
  --sum-filter random \
  --sample 10000 \
  --alpha 1.0 \
  --neg 3 \
  --window 15 \
  --epochs 1 \
  --lambda 70 \
  --sample-decay 1.9 \
  --window-decay 5 \
  --replication
```

XP005

XP105

XP006

XP106

XP007

XP107

XP008

XP108

XP009

XP109

XP010

XP110

XP011

XP111

XP012

XP112

XP013

XP113

XP014

XP114

XP015

XP115

XP016

XP116

XP017

XP117

XP018

XP118

XP019

XP119

XP020

XP120

XP021

XP121

XP022

XP122

XP023

XP123

XP024

XP124

XP025

XP125

XP026

XP126

XP027

XP127

XP028

XP128
