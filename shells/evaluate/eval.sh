#/bin/bash
LR="lr_0.0001"
python eval_result.py -pred trained_mdls/top-10/$LR.bin.dev.result -gold /home/lr/kunitaka/project/fbc/mc/data/mc_answers/train.false.top-10 --top_n 10 -out trained_mdls/top-10/$LR.bin.dev.result.eval

LR="lr_1e-05"
python eval_result.py -pred trained_mdls/top-10/$LR.bin.dev.result -gold /home/lr/kunitaka/project/fbc/mc/data/mc_answers/train.false.top-10 --top_n 10 -out trained_mdls/top-10/$LR.bin.dev.result.eval

LR="lr_3e-05"
python eval_result.py -pred trained_mdls/top-10/$LR.bin.dev.result -gold /home/lr/kunitaka/project/fbc/mc/data/mc_answers/train.false.top-10 --top_n 10 -out trained_mdls/top-10/$LR.bin.dev.result.eval