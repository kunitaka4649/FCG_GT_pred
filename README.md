# README

## USAGE

python version 3.7.13

pip 21.2.2

Download modules with pip.
```
pip install -r requirements.txt
```

Please prepare public feedback comment dataset in accordance with
[Toward a Task of Feedback Comment Generation for Writing Learning](https://aclanthology.org/D19-1316) (Nagata, EMNLP 2019)
```
root
└─data
    └─train_dev
        ├─TRAIN.prep_feedback_comment.public.tsv
        ├─DEV.prep_feedback_comment.public.tsv
        └─TEST.prep_feedback_comment.public.tsv
```

You can do experiment for top-10 grammatical term prediction using this shell script.
```
bash shells/main.sh
```

The following file will be generated after run shells/main.sh.
```
trained_mdls/top-10/lr_0.0001.bin
trained_mdls/top-10/lr_0.0001.bin.dev.result
trained_mdls/top-10/lr_0.0001.bin.dev.result.best_threshold
trained_mdls/top-10/lr_0.0001.bin.dev.result.eval
trained_mdls/top-10/lr_0.0001.bin.test.result
trained_mdls/top-10/lr_0.0001.bin.test.result.best_threshold
trained_mdls/top-10/lr_0.0001.bin.train.result
trained_mdls/top-10/lr_0.0001.bin.train.result.best_threshold
trained_mdls/top-10/lr_1e-05.bin
trained_mdls/top-10/lr_1e-05.bin.dev.result
trained_mdls/top-10/lr_1e-05.bin.dev.result.best_threshold
trained_mdls/top-10/lr_1e-05.bin.dev.result.eval
trained_mdls/top-10/lr_1e-05.bin.test.result
trained_mdls/top-10/lr_1e-05.bin.test.result.best_threshold
trained_mdls/top-10/lr_1e-05.bin.train.result
trained_mdls/top-10/lr_1e-05.bin.train.result.best_threshold
trained_mdls/top-10/lr_3e-05.bin
trained_mdls/top-10/lr_3e-05.bin.dev.result
trained_mdls/top-10/lr_3e-05.bin.dev.result.best_threshold
trained_mdls/top-10/lr_3e-05.bin.dev.result.eval
trained_mdls/top-10/lr_3e-05.bin.test.result
trained_mdls/top-10/lr_3e-05.bin.test.result.best_threshold
trained_mdls/top-10/lr_3e-05.bin.train.result
trained_mdls/top-10/lr_3e-05.bin.train.result.best_threshold
```

The files with .eval include the exact_match_ratio, hamming_loss, micro_avg_precision, micro_avg_recall, f_measure on first line of each file.
```
trained_mdls/top-10/lr_0.0001.bin.dev.result.eval
trained_mdls/top-10/lr_1e-05.bin.dev.result.eval
trained_mdls/top-10/lr_3e-05.bin.dev.result.eval
```

Among these results, the result with the highest value of f_measure is good.
In the case of seed value 1234, lr_3e-05 has the best performance.
