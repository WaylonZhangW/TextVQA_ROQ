import os
import numpy as np
import pdb
import cv2
import math
from tqdm import tqdm
import editdistance



def get_anls(s1,s2):
        s1 = s1.lower().strip()
        s2 = s2.lower().strip()
        iou = 1 - editdistance.eval(s1, s2) / max(len(s1), len(s2))
        anls = iou if iou >= .5 else 0.
        return anls


def draw_ocr_bbox(item,target_ocr_bbox,soft_label):

    img_path = item['image_path']
    textvqa = "/public/home/zhangwei1/datasets/vqa/TextVQA"
    img_path = os.path.join(textvqa,img_path)
    img = cv2.imread(img_path)
    image_height = item['image_height']
    image_width = item['image_width']
    for idx,bbox in enumerate(target_ocr_bbox):
        bbox = bbox * [image_width,image_height,image_width,image_height]
        if soft_label[idx]> 0:
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 2)

    cv2.imwrite("./a_soft_label/test_3.jpg",img)

def draw_bbox(item,target_ocr_bbox,draw_last=False):

    img_path = item['image_path']
    textvqa = "/public/home/zhangwei1/datasets/vqa/TextVQA"
    img_path = os.path.join(textvqa,img_path)
    img = cv2.imread(img_path)
    image_height = item['image_height']
    image_width = item['image_width']
    for idx,bbox in enumerate(target_ocr_bbox):
        bbox = bbox * [image_width,image_height,image_width,image_height]
        if idx == len(target_ocr_bbox)-1 and draw_last:
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (192,192,192), 2)
        else:
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (0,255,0), 1)

    cv2.imwrite("./a_soft_label/test.jpg",img)


def mkdir_if_missing(dirname):
    """Create dirname if it is missing."""
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise



def get_label():
    file_dir = "/public/home/zhangwei1/datasets/data/imdb/microsoft_ocr/microsoftOCR_spatial_train.npy"
    save_path = "/public/home/zhangwei1/datasets/data/imdb/microsoft_ocr/microsoftOCR_loss_train.npy"
    M_imdb_npy = np.load(file_dir,allow_pickle=True)
    m_imdb = M_imdb_npy[1:]
    print(type(m_imdb),M_imdb_npy[0])  

    pdb.set_trace()
    for idx,item in tqdm(enumerate(m_imdb)):
   
        ocr_tokens = item['ocr_tokens'][:50]
        ocr_tokens = [x.lower() for x in ocr_tokens]

        ocr_normalized_boxes = item['ocr_normalized_boxes']
        obj_normalized_boxes = item['obj_normalized_boxes']
        assert obj_normalized_boxes.shape == (100,4)
        gt_answer = max(item['valid_answers'],key=item['valid_answers'].count)
        # lower 
        gt_answer = gt_answer.lower()
        gt_answer = gt_answer.split(" ") # list

        obj_label = np.zeros((100), dtype=np.float32)
        gt_answer_in_ocrs = [x for x in gt_answer if x in ocr_tokens]

        # print_info(item)
        if len(gt_answer_in_ocrs)> 0:
            target_ocr_bbox = []
            for gt in gt_answer_in_ocrs:
                target_ocr_indx = ocr_tokens.index(gt)
                target_ocr_bbox.append(ocr_normalized_boxes[target_ocr_indx])

            # gt_ocr 是包括所有是答案的ocr tokens的最大bbox
            gt_ocr = target_ocr_bbox[0]
            if len(target_ocr_bbox) > 1 :
                t_x,t_y = 1.0, 1.0
                b_x,b_y = 0.0, 0.0
                for bbox in target_ocr_bbox:
                    top_left_x,top_left_y = bbox[0],bbox[1]
                    bottom_right_x,bottom_right_y = bbox[2],bbox[3]
                    if top_left_x < t_x:
                        t_x =  top_left_x
                    if top_left_y < t_y:
                        t_y = top_left_y
                    if bottom_right_x > b_x: 
                        b_x = bottom_right_x
                    if bottom_right_y > b_y: 
                        b_y = bottom_right_y
                gt_ocr[0] = t_x
                gt_ocr[1] = t_y
                gt_ocr[2] = b_x
                gt_ocr[3] = b_y
            print(type(gt_ocr),gt_ocr)
            if len(target_ocr_bbox) > 1:
                temp = target_ocr_bbox
                temp.append(gt_ocr)
                draw_bbox(item,temp,draw_last=True)
                print("done")
            else:
                draw_bbox(item,target_ocr_bbox,draw_last=False)
            
            
            # 计算每个object的bbox与 target ocr bbox的归一化后的bbox
            obj_label = Cal_object_score(gt_ocr,obj_normalized_boxes)
            print(obj_label.sum())
            
            draw_obj_bbox(item,obj_normalized_boxes,obj_label)
            # pdb.set_trace()
        m_imdb[idx]['obj_label'] = obj_label

        ### ocr label 
        ocr_label = np.zeros((50), dtype=np.float32)
        for i in range(50):
            if i >= len(ocr_tokens):
                ocr_label[i] = 0
            else:
                ocr = ocr_tokens[i]
                scores = [get_anls(ocr, gt) for gt in gt_answer]
                ocr_label[i] = max(scores)


        draw_ocr_bbox(item, ocr_normalized_boxes,ocr_label)
        print("ocr_label: ", ocr_label)    
        pdb.set_trace()

        m_imdb[idx]['ocr_label'] = ocr_label


    M_imdb_npy[1:] = m_imdb
    print(M_imdb_npy[2].keys())
    pdb.set_trace()
    # np.save(save_path,m_imdb)



def main():

    file_dir = "/public/home/zhangwei1/datasets/data/imdb/microsoft_ocr/microsoftOCR_spatial_train.npy"
    m_imdb = np.load(file_dir,allow_pickle=True)[1:]
    print(type(m_imdb))
    pdb.set_trace()
    root_path = "/public/home/zhangwei1/datasets/data/object_pseudo_gt/stvqa_soft/test_task3"  
    for idx,item in tqdm(enumerate(m_imdb)):
        
        save_path = os.path.join(root_path,item['feature_path'])
        ocr_tokens = item['ocr_tokens'][:50]
        ocr_tokens = [x.lower() for x in ocr_tokens]

        ocr_normalized_boxes = item['ocr_normalized_boxes']
        obj_normalized_boxes = item['obj_normalized_boxes']
        assert obj_normalized_boxes.shape == (100,4)
        gt_answer = max(item['valid_answers'],key=item['valid_answers'].count)
        # lower 
        gt_answer = gt_answer.lower()
        gt_answer = gt_answer.split(" ") # list
        soft_label = np.zeros((100), dtype=np.float32)
        gt_answer_in_ocrs = [x for x in gt_answer if x in ocr_tokens]


        if len(gt_answer_in_ocrs)==0 :
            m_imdb[idx]['obj_label'] = soft_label
            # np.save(save_path,soft_label)
        else:
            target_ocr_bbox = []
            for gt in gt_answer_in_ocrs:
                target_ocr_indx = ocr_tokens.index(gt)
                target_ocr_bbox.append(ocr_normalized_boxes[target_ocr_indx])


            # gt_ocr 是包括所有是答案的ocr tokens的最大bbox
            gt_ocr = target_ocr_bbox[0]
            if len(target_ocr_bbox) > 1 :
                t_x,t_y = 1.0, 1.0
                b_x,b_y = 0.0, 0.0
                for bbox in target_ocr_bbox:
                    top_left_x,top_left_y = bbox[0],bbox[1]
                    bottom_right_x,bottom_right_y = bbox[2],bbox[3]
                    if top_left_x < t_x:
                        t_x =  top_left_x
                    if top_left_y < t_y:
                        t_y = top_left_y
                    if bottom_right_x > b_x: 
                        b_x = bottom_right_x
                    if bottom_right_y > b_y: 
                        b_y = bottom_right_y
                gt_ocr[0] = t_x
                gt_ocr[1] = t_y
                gt_ocr[2] = b_x
                gt_ocr[3] = b_y
            # print(type(gt_ocr),gt_ocr)
            
            # if len(target_ocr_bbox) > 1:
            #     temp = target_ocr_bbox
            #     temp.append(gt_ocr)
            #     draw_bbox(item,temp,draw_last=True)
            #     print("done")
            # else:
            #     draw_bbox(item,target_ocr_bbox,draw_last=False)
            
            # print_info(item)
            # 计算每个object的bbox与 target ocr bbox的归一化后的bbox
            soft_label = Cal_object_score(gt_ocr,obj_normalized_boxes)
            m_imdb[idx]['obj_label'] = soft_label
            # print(soft_label.sum())
            
            # m_imdb[idx]['soft_label'] = soft_label
            # draw_obj_bbox(item,obj_normalized_boxes,soft_label)
            # pdb.set_trace()
        # print(save_path)
        # pdb.set_trace()
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # np.save(save_path,soft_label)
    save_path = "/public/home/zhangwei1/datasets/data/imdb/microsoft_ocr/microsoftOCR_temp_train.npy"
    np.save(save_path,m_imdb)


            
def draw_obj_bbox(item,obj_normalized_boxes,soft_label):


    img_path = "./a_soft_label/test.jpg"
    img = cv2.imread(img_path)
    image_height = item['image_height']
    image_width = item['image_width']
    for idx,bbox in enumerate(obj_normalized_boxes):

        bbox = bbox * [image_width,image_height,image_width,image_height]
        if soft_label[idx]!=0:
            cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,255,0), 1)

    cv2.imwrite("./a_soft_label/test2.jpg",img)
        

def Cal_object_score(ocr_bbox,object_bbox,t=0.4):
    soft_label = np.zeros((100), dtype=np.float32)
    for idx,obj in enumerate(object_bbox):
        ocr_area = (ocr_bbox[2]-ocr_bbox[0]) * (ocr_bbox[3]-ocr_bbox[1])
        obj_area = (obj[2]-obj[0]) * (obj[3]-obj[1])
        # 计算 overlap 面积
        overlap_aera = cal_overlap(obj,ocr_bbox)
        score = overlap_aera /obj_area
        if score >= t and obj_area >=  0.8 * ocr_area:
            soft_label[idx] = score
    if  soft_label.max(0) > 0 :
        soft_label = soft_label / soft_label.max(0)        
    # print(soft_label)
    return soft_label

    

def cal_overlap(obj,ocr):
    xmin = max(obj[0],ocr[0])
    ymin = max(obj[1],ocr[1])
    xmax = min(obj[2],ocr[2])
    ymax = min(obj[3],ocr[3])

    w = xmax - xmin
    h = ymax - ymin

    if w<= 0 or h <= 0:
        return 0
    else:
        return w * h

def print_info(item):
    print("question: ",item['question'])
    print("answer:",item['valid_answers'])
    print("ocr: ",item['ocr_tokens'])
    print("image class: ",item['image_classes'])

def extract_ocr_label():

    file_dir = "/public/home/zhangwei1/datasets/data/imdb/microsoft_ocr/microsoftOCR_temp_train.npy"
    m_imdb = np.load(file_dir,allow_pickle=True)[1:]
    print(type(m_imdb))
    pdb.set_trace()
    root_path = "/public/home/zhangwei1/datasets/data/ocr_pseudo_gt/textvqa_soft/train"  
    for idx,item in tqdm(enumerate(m_imdb)):
        
        save_path = os.path.join(root_path,item['feature_path'])
        ocr_tokens = item['ocr_tokens'][:50]
        ocr_tokens = [x.lower() for x in ocr_tokens]

        ocr_normalized_boxes = item['ocr_normalized_boxes']

        gt_answer = max(item['valid_answers'],key=item['valid_answers'].count)
        # lower 
        gt_answer = gt_answer.lower()
        gt_answer = gt_answer.split(" ") # list
        soft_label = np.zeros((50), dtype=np.float32)
        for idx in range(50):
            if idx >= len(ocr_tokens):
                soft_label[idx] = 0
            else:
                ocr = ocr_tokens[idx]
                scores = [get_anls(ocr, gt) for gt in gt_answer]
                soft_label[idx] = max(scores)

        # print_info(item)
        # draw_ocr_bbox(item, ocr_normalized_boxes,soft_label)
        # print("soft label: ", soft_label)    
        # pdb.set_trace()
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # np.save(save_path,soft_label)
        m_imdb[idx]['ocr_label'] = soft_label
    save_path = "/public/home/zhangwei1/datasets/data/imdb/microsoft_ocr/microsoftOCR_loss_train.npy"
    np.save(save_path,m_imdb)

if __name__ == "__main__":
    # main()
    # get_label()

    # extract_ocr_label()
    save_path = "/public/home/zhangwei1/TQD/AAA_test/imdb_noempty_train.npy"
    m_imdb = np.load(save_path,allow_pickle=True)[1:]
    print(m_imdb[2].keys())
    pdb.set_trace() 
    for idx,item in tqdm(enumerate(m_imdb)):
        print(item['ocr_tokens'])
        print(item['flickr_original_url'])
        pdb.set_trace()