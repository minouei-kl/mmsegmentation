#!/bin/bash
pip install mmcv-full==1.2.5
pip install git+file:///netscratch/minouei/report/mmsegmentation
python -u tools/train.py configs/publay/danet_r50-d8_512x512_ver3.py  --launcher=slurm

