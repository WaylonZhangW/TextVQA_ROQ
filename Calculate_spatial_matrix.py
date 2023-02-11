import torch
import numpy as np
import os
from torch import nn

########################## 判断两个bbox是否相交 ###########################
def Intersect(bbox_1,bbox_2):
    # 2个bbox中心的距离
    mid_x = abs( (bbox_1[0] + bbox_1[2]) / 2 - (bbox_2[0] + bbox_2[2])/2 )
    mid_y = abs( (bbox_1[1] + bbox_1[3]) / 2 - (bbox_2[1] + bbox_2[3])/2 )

    # 2个bbox 边长之和
    width = abs(bbox_1[2] - bbox_1[0]) + abs(bbox_2[2] - bbox_1[0])
    height = abs(bbox_1[3] - bbox_1[1]) + abs(bbox_2[3] - bbox_1[1])

    if (mid_x <= width/2 and mid_y<= height/2):
        return True

    return False
###########################################################

#### 计算两点之间的曼哈顿距离   ########
def Manhattan_distance(x1,y1,x2,y2):
    d = abs(x1-x2) + abs(y1-y2)
    return d

########################## 计算距离矩阵  #################################

def cal_distance_matrix():
    """计算每个ocr bbox的距离 """
    file_dir = '/root/CS282/data/imdb/imdb_sbd/0510_train_addObjLabel_Box.npy'
    print(file_dir)
    imdb = np.load(file_dir,allow_pickle=True)

    print('begin calculate...')

    for idx,item in enumerate(imdb):
        if idx==0:
            continue

        # ocr_bbox = imdb[idx]['ocr_normalized_boxes']
        ocr_info = imdb[idx]['ocr_info']
        assert len(ocr_info) == len(imdb[idx]['ocr_tokens'])
        num_bbox = min(len(ocr_info),50)
        ocr_bbox = np.zeros((50,4), dtype=np.float32)
        for i,ocr in enumerate(ocr_info[:num_bbox]):
            bbox = ocr['bounding_box']
            x = bbox["top_left_x"]
            y = bbox["top_left_y"]
            width = bbox["width"]
            height = bbox["height"]

            ocr_bbox[i][0] = x
            ocr_bbox[i][1] = y
            ocr_bbox[i][2] = x + width
            ocr_bbox[i][3] = y + height

        # imdb[idx]['ocr_normalized_boxes'] = ocr_bbox

        Distance = np.zeros((50, 50), dtype=np.float32)
        for i in range(num_bbox):
            for j in range(num_bbox):
                A_ij = calculate_A(ocr_bbox[i],ocr_bbox[j])
                Distance[i][j] = 1 - A_ij
        # norm
        imdb[idx]['Distance_Matrix'] = Distance

    save_dir = "/root/CS282/data/imdb/new_textvqa/textvqa_imdb_train.npy"
    print(save_dir)
    np.save(save_dir,imdb)
    print('Done')

def calculate_A(bbox_1,bbox_2):
    # 如果相交，距离为0
    if Intersect(bbox_1,bbox_2):
        return 0
    L = []
    #   不相交，最小距离为 顶点之间的距离（共16种情况，求其最小即可）
    # 1
    L.append(Manhattan_distance(bbox_1[0], bbox_1[1], bbox_2[0], bbox_2[1]) )
    L.append(Manhattan_distance(bbox_1[0], bbox_1[1], bbox_2[0], bbox_2[3]))
    L.append(Manhattan_distance(bbox_1[0], bbox_1[1], bbox_2[2], bbox_2[1]))
    L.append(Manhattan_distance(bbox_1[0], bbox_1[1], bbox_2[2], bbox_2[3]))
    # 2
    L.append(Manhattan_distance(bbox_1[0], bbox_1[3], bbox_2[0], bbox_2[1]) )
    L.append(Manhattan_distance(bbox_1[0], bbox_1[3], bbox_2[0], bbox_2[3]))
    L.append(Manhattan_distance(bbox_1[0], bbox_1[3], bbox_2[2], bbox_2[1]))
    L.append(Manhattan_distance(bbox_1[0], bbox_1[3], bbox_2[2], bbox_2[3]))
    # 3
    L.append(Manhattan_distance(bbox_1[2], bbox_1[1], bbox_2[0], bbox_2[1]))
    L.append(Manhattan_distance(bbox_1[2], bbox_1[1], bbox_2[0], bbox_2[3]))
    L.append(Manhattan_distance(bbox_1[2], bbox_1[1], bbox_2[2], bbox_2[1]))
    L.append(Manhattan_distance(bbox_1[2], bbox_1[1], bbox_2[2], bbox_2[3]))
    # 4
    L.append(Manhattan_distance(bbox_1[2], bbox_1[3], bbox_2[0], bbox_2[1]))
    L.append(Manhattan_distance(bbox_1[2], bbox_1[3], bbox_2[0], bbox_2[3]))
    L.append(Manhattan_distance(bbox_1[2], bbox_1[3], bbox_2[2], bbox_2[1]))
    L.append(Manhattan_distance(bbox_1[2], bbox_1[3], bbox_2[2], bbox_2[3]))

    distance = min(L)

    return min(distance,1)


if __name__ == "__main__":
    cal_distance_matrix()

