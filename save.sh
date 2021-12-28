#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 main.py 2>&1|tee ./log/sceneflow_gam.txt



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 finetune.py 2>&1|tee ./log/kitti_gam.txt

