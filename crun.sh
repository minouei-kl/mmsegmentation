#!/bin/bash
pip install mmcv-full==1.2.5
pip install git+file:///netscratch/minouei/report/mmsegmentation
python -u tools/train.py configs/publay/ccnet_r50-d8_512x512_80k_ver5.py  --launcher=slurm

