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
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
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
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
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
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
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
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
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
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
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
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --reload
```

XP105
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only
```

XP006
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter random \
  --sample 10000 \
  --reload
```

XP106
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter random \
  --sample 10000
```

XP007
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter self \
  --sum-threshold 22 \
  --reload
```

XP107
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter self \
  --sum-threshold 22
```

XP008
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter cwi \
  --sum-threshold 0 \
  --reload
```

XP108
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter cwi \
  --sum-threshold 0
```

XP009
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt \
  --sum-only
```

XP010
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt \
  --sum-only \
  --sum-filter random \
  --sample 10000
```


XP011
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt \
  --sum-only \
  --sum-filter self \
  --sum-threshold 22
```


XP012
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter cwi \
  --sum-threshold 0
```


XP013
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt \
  --sum-only
```

XP014
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt \
  --sum-only \
  --sum-filter random \
  --sample 10000
  ```


XP015
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt \
  --sum-only \
  --sum-filter self \
  --sum-threshold 22
```


XP016
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter cwi \
  --sum-threshold 0
```


XP017
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt \
  --sum-only
```

XP018
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt \
  --sum-only \
  --sum-filter random \
  --sample 10000
```

XP019
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt \
  --sum-only \
  --sum-filter self \
  --sum-threshold 22
```

XP020
```
/home/kabbach/venv/bin/n2v test \
  --on chimeras \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-only \
  --sum-filter cwi \
  --sum-threshold 0
```

XP021
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-filter cwi \
  --sum-threshold 0 \
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

XP121
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-filter cwi \
  --sum-threshold 0 \
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

XP022
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-filter cwi \
  --sum-threshold 0 \
  --train-with cwi_alpha \
  --alpha 1.0 \
  --beta 1000 \
  --kappa 1 \
  --neg 3 \
  --epochs 1 \
  --reload
```

XP122
```
/home/kabbach/venv/bin/n2v test \
  --on definitions \
  --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test \
  --model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --info-model /home/kabbach/nonce2vec/models/enwiki.20190120.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model \
  --sum-filter cwi \
  --sum-threshold 0 \
  --train-with cwi_alpha \
  --alpha 1.0 \
  --beta 1000 \
  --kappa 1 \
  --neg 3 \
  --epochs 1
```

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
