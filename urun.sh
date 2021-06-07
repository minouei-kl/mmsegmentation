#!/bin/bash
pip install mmcv-full==1.2.5
pip install git+file:///netscratch/minouei/report/mmsegmentation
python -u tools/train.py configs/publay/deeplabv3_unet_s5-d16_64x64_40k_ver.py  --launcher=slurm
