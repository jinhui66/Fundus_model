from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2

def convert_coco2mask():
    catIds = coco.getCatIds()
    imgIds = coco.getImgIds(catIds=catIds) # 获取所有图片的id
    print("Total images:", len(imgIds))
    for image_id in imgIds:
        img = coco.imgs[image_id]
        image = np.array(Image.open(os.path.join(img_dir, img['file_name'])))
        print(img['file_name'])

        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        coco.showAnns(anns)
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        # cv2.imwrite(os.path.join(save_dir, "mask{}.png".format(image_id)), mask)
        cv2.imwrite(os.path.join(save_dir, img['file_name']), mask)


if __name__ == '__main__':
    Dataset_dir = "F:/AAA-projects/ING/3-march/700-tooth/x-ray/training_data/training_data/quadrant_enumeration/"
    coco = COCO(os.path.join(Dataset_dir, 'train_quadrant_enumeration.json'))
    img_dir = os.path.join(Dataset_dir, 'xrays')
    save_dir = os.path.join(Dataset_dir, "Mask")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    convert_coco2mask()
