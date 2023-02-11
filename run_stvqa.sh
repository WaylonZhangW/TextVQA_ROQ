#!/bin/bash
source activate
conda activate tqd 
## microsoft OCR on ST-VQA
CUDA_VISIBLE_DEVICES=2 python tools/run.py --tasks vqa --datasets tqd_stvqa --model tqd_2 \
--config configs/tqd/tqd_stvqa/tqd_2_microsoft.yml \
--save_dir hand_over/STVQA training_parameters.seed 13

