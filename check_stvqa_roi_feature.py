import torch
import  numpy as np
from torch import nn

def run():
    file_dir = '/root/CS282/data/our_stvqa_ocr_roi_feature_512/ST-VQA/coco-text/COCO_train2014_000000000142.npy'
    imdb = np.load(file_dir, allow_pickle=True)
    print(imdb.shape,imdb)

    file_dir_1 = '/root/CS282/data/stvqa_ocr_frcn_features/ST-VQA/coco-text/COCO_train2014_000000000142.npy'
    imdb1 = np.load(file_dir_1, allow_pickle=True)
    print(imdb1.shape)
    file_dir_2 = '/root/CS282/data/stvqa_ocr_frcn_features/ST-VQA/coco-text/COCO_train2014_000000000142_info.npy'
    imdb2 = np.load(file_dir_2, allow_pickle=True)
    print(imdb2)

    # assert  False

    dir_2 = '/root/CS282/data/our_stvqa_ocr_frcn_features/ST-VQA/coco-text/COCO_train2014_000000000142.npy'
    imdb3 = np.load(dir_2, allow_pickle=True)
    x2 = torch.from_numpy(imdb3)
    # pooling = nn.AvgPool2d(kernel_size=7)
    # x2 = pooling(x2)
    print(imdb3.shape)
if __name__ == "__main__":
    # run()
    file_dir = '/root/CS282/data/imdb/our_stvqa/stvqa_imdb_test_task3.npy'
    imdb = np.load(file_dir, allow_pickle=True)
    print(len(imdb))
    print(imdb[:2])

    # print(imdb[2])
    # for idx,info in enumerate(imdb):
    #     image_name = info['image_name']
    #     if image_name == 'COCO_train2014_000000001090.jpg':
    #             print(info)
    #             break

