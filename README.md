<<<<<<< HEAD

# Guiding Scene Text Question Answering with Region Priors
本代码是基于[Ssbaseline](https://github.com/ZephyrZhuQi/ssbaseline) 和 [M4C](https://github.com/ronghanghu/mmf/tree/project/m4c_captioner_pre_release/projects/M4C)。
## 改动
其中在Evaluation的时候，原代码有一个bug。原代码的算法逻辑是，每个batch计算vqa accuracy，然后n个batch得到n个数值。 就是如果总样本数不能被batch size整除的话，最后一个batch中的样本是少于batch size的，而计算的时候它和之前的值是平等对待的。所以会出现结果随着batch size的大小改变的bug。  
现在的逻辑改成了计算每个batch中样本的vqa accuracy之和，相加后除以总共的样本数。

## Installation
以下实验都是基于**PyTorch 1.8.1**, **CUDA 10.2 或者 CUDA 11.1**.
```
python setup.py build develop
```
## Data
对于ST-VQA：原数据中coco的图片被resize成了224*224（在原论文中有提及），所以提取image feature可能会不准确，我把coco的图片都替换成了未被resize的原本的图片。  
之前的模型都是使用faster rcnn的 fc6层后的feature（2048维），为了后面的聚类，我是提取了roi之后，maxpooling后的feature（512维），采用的提取feature的框架是detectron2，和M4C的一致，脚本是`Calculate_spatial_matrix.py`，但是还需要更改detectron2的源代码使得输出的是roi的feature。  
同时我们还使用了TAP模型mmt层之后的feature。

```
Calculate_spatial_matrix.py: 计算每个OCR tokens之间的曼哈顿距离。
Get_soft.py: 计算得到答案所在区域的伪标签，具体流程参考论文。
```

## Training and Evaluation
1. Train the TQD model on the TextVQA training set.
```shell
bash run_textvqa.sh
```
2. Train the TQD model on the ST training set.
```shell
bash run_stvqa.sh
```

2. Evaluate the pretained TQD model locally on the TextVQA validation set.
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






