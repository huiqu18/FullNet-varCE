#!/usr/bin/env bash
exp='1'
gpus='01'
epoch='best'
python train.py --alpha 1.0 --batch-size 8 --lr 0.0005 --epochs 1000 --gpu ${gpus} \
     --save-dir ./experiments/GlaS/${exp}

python test.py --model-path ./experiments/GlaS/${exp}/checkpoints/checkpoint_${epoch}.pth.tar \
    --img-dir ./data/GlaS/images/testA --label-dir ./data/GlaS/labels_instance/testA \
    --save-dir ./experiments/GlaS/${exp}/${epoch} --gpu ${gpus}

python test.py --model-path ./experiments/GlaS/${exp}/checkpoints/checkpoint_${epoch}.pth.tar \
    --img-dir ./data/GlaS/images/testB --label-dir ./data/GlaS/labels_instance/testB \
    --save-dir ./experiments/GlaS/${exp}/${epoch} --gpu ${gpus}
