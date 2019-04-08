# XP summary

## Replication
Training w2v background model on Wikipedia dump:
```
/home/kabbach/venv/bin/n2v train \
  --data /home/kabbach/witokit/data/wiki/enwiki.20190120.txt
  --outputdir /home/kabbach/nonce2vec/models/
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads 27
  --train-mode skipgram
```

## Informativeness
Training informativeness background model with same hyperparameters as background model:
```
/home/kabbach/venv/bin/n2v train \
  --data /home/kabbach/witokit/data/wiki/enwiki.20190120.txt
  --outputdir /home/kabbach/nonce2vec/models/
  --alpha 0.025 \
  --neg 5 \
  --window 5 \
  --sample 1e-3 \
  --epochs 5 \
  --min-count 50 \
  --size 400 \
  --num-threads 27
  --train-mode cbow
```
