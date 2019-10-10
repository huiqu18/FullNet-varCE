#!/usr/bin/env bash
exp='1'
gpus='23'
python train.py --alpha 1.0 --batch-size 8 --lr 0.001 --epochs 300 --gpu ${gpus} \
     --save-dir ./experiments/MultiOrgan/${exp}

python test.py --model-path ./experiments/MultiOrgan/${exp}/checkpoints/checkpoint_best.pth.tar \
    --img-dir ./data/MultiOrgan/images/test_same --label-dir ./data/MultiOrgan/labels_instance/test \
    --save-dir ./experiments/MultiOrgan/${exp}/best --gpu ${gpus}

python test.py --model-path ./experiments/MultiOrgan/${exp}/checkpoints/checkpoint_best.pth.tar \
    --img-dir ./data/MultiOrgan/images/test_diff --label-dir ./data/MultiOrgan/labels_instance/test \
    --save-dir ./experiments/MultiOrgan/${exp}/best  --gpu ${gpus}
