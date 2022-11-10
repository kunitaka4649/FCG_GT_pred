#!/bin/bash

bash shells/train/top-10.sh

bash shells/predict/lr_0.0001_top-10.sh
bash shells/predict/lr_1e-05_top-10.sh
bash shells/predict/lr_3e-05_top-10.sh

bash shells/evaluate/eval.sh