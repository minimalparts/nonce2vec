# XP CMD

## XP005
```
n2v train --data /home/kabbach/nonce2vec/data/enwiki.20180920.utf8.lower.txt --outputdir /home/kabbach/nonce2vec/models/ --alpha 0.025 --neg 5 --window 5 --sample 1e-3 --epochs 5 --min_count 50 --size 400 --train_mode skipgram --num_threads 15
```

```
n2v check --data /home/kabbach/nonce2vec/data/MEN/MEN_dataset_natural_form_full --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model
```

## XP006
```
n2v train --data /home/kabbach/nonce2vec/data/enwiki.20180920.utf8.lower.txt --outputdir /home/kabbach/nonce2vec/models/ --alpha 0.025 --neg 5 --window 5 --sample 1e-3 --epochs 5 --min_count 50 --size 700 --train_mode skipgram --num_threads 15
```

```
n2v check --data /home/kabbach/nonce2vec/data/MEN/MEN_dataset_natural_form_full --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size700.model
```

## XP007
```
n2v train --data /home/kabbach/nonce2vec/data/enwiki.20180920.utf8.lower.txt --outputdir /home/kabbach/nonce2vec/models/ --alpha 0.025 --neg 5 --window 5 --sample 1e-3 --epochs 5 --min_count 50 --size 400 --train_mode cbow --num_threads 10
```

```
n2v check --data /home/kabbach/nonce2vec/data/MEN/MEN_dataset_natural_form_full --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model
```

## XP008
```
n2v train --data /home/kabbach/nonce2vec/data/enwiki.20180920.utf8.lower.txt --outputdir /home/kabbach/nonce2vec/models/ --alpha 0.010 --neg 10 --window 50 --sample 1e-3 --epochs 5 --min_count 50 --size 1000 --train_mode cbow --num_threads 10
```

```
n2v check --data /home/kabbach/nonce2vec/data/MEN/MEN_dataset_natural_form_full --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.010.neg10.win50.sample0.001.epochs5.mincount50.size1000.model
```

## XP009
```

```

## XP010
```

```

## XP011
```

```
