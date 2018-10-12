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
n2v check --data /home/kabbach/nonce2vec/data/MEN/MEN_dataset_natural_form_full --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.01.neg10.win50.sample0.001.epochs5.mincount50.size1000.model
```

## XP009
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP010
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP011
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP014
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set
```

## XP015
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter random --sample 10000
```

## XP016
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter self --sum_threshold 22
```

## XP017
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter cwi --sum_threshold 0 --info_model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model
```

## XP117
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter cwi --sum_threshold 0 --info_model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.cbow.alpha0.01.neg10.win50.sample0.001.epochs5.mincount50.size1000.model
```

## XP018
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set
```

## XP019
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter random --sample 10000
```

## XP020
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter self --sum_threshold 22
```

## XP021
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter cwi --sum_threshold 0 --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model
```

## XP121
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_only --sum_over_set --sum_filter cwi --sum_threshold 0 --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.01.neg10.win50.sample0.001.epochs5.mincount50.size1000.model
```

## XP023
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter random --sample 10000 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication
```

## XP024
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter cwi --sum_threshold 0 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication
```

## XP124
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.cbow.alpha0.01.neg10.win50.sample0.001.epochs5.mincount50.size1000.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter cwi --sum_threshold 0 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication
```

## XP025
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_filter cwi --sum_threshold 0 --sum_over_set --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --train_with cwi_alpha --alpha 1 --beta 1000 --kappa 1
```

## XP125
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.cbow.alpha0.01.neg10.win50.sample0.001.epochs5.mincount50.size1000.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_filter cwi --sum_threshold 0 --sum_over_set --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --train_with cwi_alpha --alpha 1 --beta 1000 --kappa 1
```

## XP026

## XP027
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter random --sample 10000 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication
```

## XP028
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter cwi --sum_threshold 0 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication
```

## XP128
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.01.neg10.win50.sample0.001.epochs5.mincount50.size1000.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter cwi --sum_threshold 0 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication
```

## XP029
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter cwi --sum_threshold 0 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --train_with cwi_alpha --alpha 1 --beta 1000 --kappa 1
```

## XP129
```
n2v test --on nonces --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.01.neg10.win50.sample0.001.epochs5.mincount50.size1000.model --data /home/kabbach/nonce2vec/data/definitions/nonce.definitions.299.test --sum_over_set --sum_filter cwi --sum_threshold 0 --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --train_with cwi_alpha --alpha 1 --beta 1000 --kappa 1
```

## XP038
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP039
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP040
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP043
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP044
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP045
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP048
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/wiki_all.sent.split.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP049
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/wiki.all.utf8.sent.split.lower.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP050
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --alpha 1 --neg 3 --window 15 --sample 10000 --epochs 1 --lambda 70 --sample_decay 1.9 --window_decay 5 --replication --sum_filter random --sum_over_set
```

## XP051
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --sum_only --sum_over_set
```

## XP052
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --sum_only --sum_over_set --sum_filter random --sample 10000
```

## XP053
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --sum_only --sum_over_set --sum_filter self --sum_threshold 22
```

## XP054
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l2.tokenised.test.txt --sum_only --sum_over_set --sum_filter cwi --sum_threshold 0 --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model
```

## XP055
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt --sum_only --sum_over_set
```

## XP056
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt --sum_only --sum_over_set --sum_filter random --sample 10000
```

## XP057
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt --sum_only --sum_over_set --sum_filter self --sum_threshold 22
```

## XP058
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l4.tokenised.test.txt --sum_only --sum_over_set --sum_filter cwi --sum_threshold 0 --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model
```

## XP059
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --sum_only --sum_over_set
```

## XP060
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --sum_only --sum_over_set --sum_filter random --sample 10000
```

## XP061
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --sum_only --sum_over_set --sum_filter self --sum_threshold 22
```

## XP062
```
n2v test --on chimeras --model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.skipgram.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model --data /home/kabbach/nonce2vec/data/chimeras/chimeras.dataset.l6.tokenised.test.txt --sum_only --sum_over_set --sum_filter cwi --sum_threshold 0 --info_model /home/kabbach/nonce2vec/models/enwiki.20180920.utf8.lower.txt.cbow.alpha0.025.neg5.win5.sample0.001.epochs5.mincount50.size400.model
```
