#!/bin/bash
source activate
conda activate tqd 
# Microsoft OCR 
CUDA_VISIBLE_DEVICES=1 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_2 --config configs/tqd/tqd_2_bert.yml \
--save_dir hand_over/microsoft_ocr_Wo_stvqa training_parameters.seed 13

# Microsoft OCR with ST-VQA
CUDA_VISIBLE_DEVICES=1 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_2 --config configs/tqd/tqd_2_bert_with_stvqa.yml \
--save_dir hand_over/microsoft_ocr_w_stvqa training_parameters.seed 13

# Rosetta-en OCR
CUDA_VISIBLE_DEVICES=1 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_2 --config configs/tqd/tqd_2_bert_rosseta.yml \
--save_dir hand_over/rosseta_ocr_wo_stvqa training_parameters.seed 13

# SBD-Trans OCR
CUDA_VISIBLE_DEVICES=1 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_2_sbd --config configs/tqd/tqd_2_bert_rosseta.yml \
--save_dir hand_over/sbd_ocr_wo_stvqa training_parameters.seed 13

# SBD-Trans OCR with ST-VQA
CUDA_VISIBLE_DEVICES=1 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_2_sbd --config configs/tqd/tqd_2_bert_sbd_with_stvqa.yml \
--save_dir hand_over/sbd_ocr_w_stvqa training_parameters.seed 13


# Microsoft OCR with Pre-training
CUDA_VISIBLE_DEVICES=1 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_pretrain2 --config configs/tqd/tqd_pretrain_textvqa/tqd_microsoft_pretrain_tap_2.yml \
--save_dir hand_over/with_pretrain training_parameters.seed 13 training_parameters.batch_size 100
