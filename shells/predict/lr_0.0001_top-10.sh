#/bin/bash
LR=0.0001
OUT_PATH=trained_mdls/top-10/lr_$LR.bin.train.result

python mc_pred.py --model_path trained_mdls/top-10/lr_$LR.bin --out_path $OUT_PATH --top_n 10 --class_weight_pkl data/grammar_terms/idf_weight_of_top-10.pkl --dev_set data/train_dev/TRAIN.prep_feedback_comment.public.tsv

THRESH=`cat $OUT_PATH.best_threshold`

python mc_pred.py --model_path trained_mdls/top-10/lr_$LR.bin --out_path trained_mdls/top-10/lr_$LR.bin.dev.result --top_n 10 --class_weight_pkl data/grammar_terms/idf_weight_of_top-10.pkl --dev_set data/train_dev/DEV.prep_feedback_comment.public.tsv --thresh $THRESH

python mc_pred.py --model_path trained_mdls/top-10/lr_$LR.bin --out_path trained_mdls/top-10/lr_$LR.bin.test.result --top_n 10 --class_weight_pkl data/grammar_terms/idf_weight_of_top-10.pkl --dev_set data/train_dev/TEST.prep_feedback_comment.public.tsv --thresh $THRESH --is_test