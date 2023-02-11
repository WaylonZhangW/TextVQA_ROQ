import os
import numpy as np
import tqdm
import argparse
import torch
from PIL import Image
import cv2
from torchsummary import summary

# install `vqa-maskrcnn-benchmark` from
# https://github.com/ronghanghu/vqa-maskrcnn-benchmark-m4c
import sys; sys.path.append('/p300/maskrcnn')  # NoQA
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict


def load_detection_model(yaml_file, yaml_ckpt):
    cfg.merge_from_file(yaml_file)
    cfg.freeze()

    model = build_detection_model(cfg)
   # summary(model,input_size=[(800,1088),(7,4)])
    checkpoint = torch.load(yaml_ckpt, map_location=torch.device("cpu"))

    load_state_dict(model, checkpoint.pop("model"))

    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    return model


def _image_transform(image_path):
    img = Image.open(image_path)
    im = np.array(img).astype(np.float32)
    # handle a few corner cases
    if im.ndim == 2:  # gray => RGB
        im = np.tile(im[:, :, None], (1, 1, 3))
    if im.shape[2] > 3:  # RGBA => RGB
        im = im[:, :, :3]

    im = im[:, :, ::-1]  # RGB => BGR
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)
    return img, im_scale


def _process_feature_extraction(
    output, im_scales, feat_name='roi_ave_pooling'
):
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    feat_list = []
    bbox_list = []

    for i in range(batch_size):
        dets = output[0]["proposals"][i].bbox / im_scales[i]
        scores = score_list[i]

        max_conf = torch.zeros((scores.shape[0])).to(cur_device)

        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)
            max_conf[keep] = torch.where(
                cls_scores[keep] > max_conf[keep],
                cls_scores[keep],
                max_conf[keep]
            )

        keep_boxes = torch.argsort(max_conf, descending=True)[:100]
        feat_list.append(feats[i][keep_boxes])
        bbox_list.append(output[0]["proposals"][i].bbox[keep_boxes])
    return feat_list, bbox_list


def extract_features(
    detection_model, image_path, input_boxes=None, feat_name='roi_ave_pooling'
):
    im, im_scale = _image_transform(image_path)
    if input_boxes is not None:
        if isinstance(input_boxes, np.ndarray):
            input_boxes = torch.from_numpy(input_boxes.copy())
        input_boxes *= im_scale

    img_tensor, im_scales = [im], [im_scale]
    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        output = detection_model(
            current_img_list, input_boxes=input_boxes)

    if input_boxes is None:
        feat_list, bbox_list = _process_feature_extraction(
            output, im_scales, feat_name)
        feat = feat_list[0].cpu().numpy()
        bbox = bbox_list[0].cpu().numpy() / im_scale
    else:
        feat = output[0][feat_name].cpu().numpy()
        bbox = output[0]["proposals"][0].bbox.cpu().numpy() / im_scale

    return feat, bbox


def get_ocr_bbox(ocr_info):
    '''imdb文件中没有ocr_normalized_boxes，从ocr_info中计算得到'''
    ocr_num = len(ocr_info)
    ocr_bbox = np.zeros((ocr_num,4),np.float32)
    for idx, info in enumerate(ocr_info):
        bbox = info["bounding_box"]
        x = bbox["top_left_x"]
        y = bbox["top_left_y"]
        width = bbox["width"]
        height = bbox["height"]

        ocr_bbox[idx][0] = x
        ocr_bbox[idx][1] = y
        ocr_bbox[idx][2] = x + width
        ocr_bbox[idx][3] = y + height

    return ocr_bbox

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detection_cfg", type=str,
        default='/p300/maskrcnn/data/detectron_model.yaml',
        help="Detectron config file; download it from https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.yaml"
    )
    parser.add_argument(
        "--detection_model", type=str,
        default='/p300/maskrcnn/data/detectron_model.pth',
        help="Detectron model file; download it from https://dl.fbaipublicfiles.com/pythia/detectron_model/detectron_model.pth"
    )
    parser.add_argument(
        "--imdb_file", type=str,
        default='/root/CS282/data/imdb/ssbaseline_stvqa/qi_obj_frcn_imdb_test_task3.npy',
        help="The imdb to extract features"
    )
    parser.add_argument(
        "--image_dir", type=str,
        default='/p300/datasets/ST-VQA/test_task3_imgs',
        help="The directory containing images"
    )
    parser.add_argument(
        "--save_dir", type=str,
        default='/root/CS282/data/new_stvqa_ocr_roi_feature_512/test_task3_imgs',
        help="The directory to save extracted features"
    )
    args = parser.parse_args()

    DETECTION_YAML = args.detection_cfg
    DETECTION_CKPT = args.detection_model
    IMDB_FILE = args.imdb_file
    IMAGE_DIR = args.image_dir
    SAVE_DIR = args.save_dir

    imdb = np.load(IMDB_FILE, allow_pickle=True)[1:]
    # keep only one entry per image_id
    image_id2info = {info['image_id']: info for info in imdb}
    imdb = list(image_id2info[k] for k in sorted(image_id2info))

    detection_model = load_detection_model(DETECTION_YAML, DETECTION_CKPT)
    print('Faster R-CNN OCR features')
    print('\textracting from', IMDB_FILE)
    print('\tsaving to', SAVE_DIR)
    for n, info in enumerate(tqdm.tqdm(imdb)):
        image_path = info['image_path']
        # print('image_path00:', image_path)
        image_path = os.path.join(IMAGE_DIR, image_path)
        # print('image_path:',image_path)
        save_feat_path = os.path.join(SAVE_DIR, info['feature_path'])
        # print('save_feat_path',save_feat_path)
        # save_info_path = save_feat_path.replace('.npy', '_info.npy')
        os.makedirs(os.path.dirname(save_feat_path), exist_ok=True)

        w = info['image_width']
        h = info['image_height']
        if 'ocr_normalized_boxes' in info.keys():
            ocr_normalized_boxes = np.array(info['ocr_normalized_boxes'])
        else:
            ocr_normalized_boxes = get_ocr_bbox(info['ocr_info'])
        ocr_boxes = ocr_normalized_boxes.reshape(-1, 4) * [w, h, w, h]
        ocr_tokens = info['ocr_tokens']
        if len(ocr_boxes) > 0:
            extracted_feat, bbox = extract_features(
                detection_model, image_path, input_boxes=ocr_boxes
            )
        else:
            extracted_feat = np.zeros((0,512), np.float32)

        # np.save(
        #     save_info_path, {'ocr_boxes': ocr_boxes, 'ocr_tokens': ocr_tokens}
        # )


        np.save(save_feat_path, extracted_feat)


if __name__ == '__main__':
    main()
