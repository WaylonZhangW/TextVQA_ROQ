import os
import shutil



###  把ST-VQA中coco-text的图片（缩放成 256 * 256）， 换成原来版本

if __name__ == "__main__":
    old_dir =  '/p300/datasets/ST-VQA/test_task3_imgs/coco-text'
    all_image = os.listdir(old_dir)
    print(len(all_image))
    new_dir =  '/p300/datasets/ST-VQA/test_task3_imgs/coco-text-new'
    coco_dir =  '/p300/datasets/train2014'
    all_image1 = os.listdir(coco_dir)
    print(len(all_image1))

    for image in all_image:
        src = os.path.join(coco_dir,image)
        dst = os.path.join(new_dir,image)
        shutil.copyfile(src,dst)

    print('done')


