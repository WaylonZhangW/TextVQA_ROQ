
# Guiding Scene Text Question Answering with Region Priors
Here is the code for the ROQ model. Note that our code is build on the [Ssbaseline](https://github.com/ZephyrZhuQi/ssbaseline)
and [M4C](https://github.com/ronghanghu/mmf/tree/project/m4c_captioner_pre_release/projects/M4C).
Thanks a lot for their contribution!

## Installation
All experiments in the paper are based on **PyTorch 1.8.1**, **CUDA 10.2**.  
Clone this repository, and build it with the following command.
```
git clone https://github.com/WaylonZhangW/TextVQA_ROQ.git
cd ./TextVQA_ROQ
python setup.py build develop
```
## Data
### Get the feature embedding
Getting data following the previous work([Ssbaseline](https://github.com/ZephyrZhuQi/ssbaseline),
[M4C](https://github.com/ronghanghu/mmf/tree/project/m4c_captioner_pre_release/projects/M4C)
and [TAP](https://github.com/microsoft/TAP)). However, the approach of our OCR feature extraction is different from
previous work. To better perform OCR-OCR query process, we first get the 7*7*512 features after ROI pooling, then use 
MaxPooling to get the final 512-dimensional feature, which contains more visual texture information. (previous work uses 
the 2048-dimensional fc6 feature after ROI pooling, which contains more semantic information)  
If you get the ST-VQA images from the official website, there are also some difference. For ST-VQA, 
we notice that many images from COCO-Text in the downloaded ST-VQA data (around 1/3 of all images) are 
resized to 256×256 for unknown reasons, which degrades the image quality and distorts their aspect ratios. 
In the released object and OCR features below, we replaced these images with their original versions from 
COCO-Text as inputs to object detection and OCR systems. 
### Get the distance matrix between OCR tokens
We calculate the distance between each two OCR tokens, more details could be found in the paper.
```
python Calculate_spatial_matrix.py
```
### Get the soft labels
We first find the target OCR tokens by mapping the
ground-truth answers to OCR tokens’ texts, and then obtain
the minimal bounding box which covers all the target OCR
tokens. 
```
python Get_soft.py
```

## Inference bug 
其中在Evaluation的时候，原代码有一个bug。原代码的算法逻辑是，每个batch计算vqa accuracy，然后n个batch得到n个数值。如果总样本数不能被batch size整除的话，最后一个batch中的样本是少于batch size的，而计算的时候它和之前的值是平等对待的。所以会出现结果随着batch size的大小改变的bug。  
现在的逻辑改成了计算每个batch中样本的vqa accuracy之和，相加后除以总共的样本数。这个bug在
[TAP issue](https://github.com/microsoft/TAP/issues/20)中被其他研究人员指出。

## Training and Evaluation
1. Train the ROQ model on the TextVQA training set.
```shell
bash run_textvqa.sh
```
2. Train the ROQ model on the ST-VQA training set.
```shell
bash run_stvqa.sh
```

2. Evaluate the pretained ROQ model locally on the TextVQA validation set.
```shell
CUDA_VISIBLE_DEVICES=0 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_2 --config configs/tqd/tqd_2_bert.yml \
--save_dir hand_over/microsoft_ocr_Wo_stvqa \
--run_type val \
--resume_file best.ckpt

```

3. Generate the EvalAI prediction files for the TextVQA test set （训练后会自动生成EvalAI的json文件）
```shell
CUDA_VISIBLE_DEVICES=0 python tools/run.py --tasks vqa --datasets tqd_textvqa \
--model tqd_2 --config configs/tqd/tqd_2_bert.yml \
--save_dir hand_over/microsoft_ocr_Wo_stvqa \
--run_type inference --evalai_inference 1 \
--resume_file best.ckpt
```






