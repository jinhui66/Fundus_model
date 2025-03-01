import os
import imgviz
import PIL
import numpy as np
import cv2
from PIL import Image
import os.path as osp


def makesave(filename, mask):
    if os.path.splitext(filename)[1] != "png":
        filename += ".png"
    if mask.min() > -1 and mask.max() < 255:
        # = Image.fromarray()
        mask_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        mask_pil.putpalette(colormap.flattern())
        mask_pil.save(filename)

    else:
        print("error")


def data_convert_colormap(src_folder, target_folder):
    if osp.isdir(target_folder) == False:
        os.mkdir(target_folder)
    image_names = os.listdir(src_folder)
    for image_name in image_names:
        image_path = osp.join(src_folder, image_name)
        mask = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
        colormap = imgviz.label_colormap()
        # print(colormap)
        lbl_pil.putpalette(colormap.flatten())
        save_path = osp.join(target_folder, image_name)
        lbl_pil.save(save_path)
        # cv2.imwrite(save_path, img)


if __name__ == '__main__':
    # red是第一个类 green是第二个类
    # data_convert_colormap(src_folder=r"F:\AAA-projects\ING\3-march\700-tooth\4_split_dataset\Training_Labels",
    #              target_folder=r"F:\AAA-projects\ING\3-march\700-tooth\4_split_dataset\Training_Labels_colormap")
    # 查看颜色变化
    colormap = imgviz.label_colormap()
    print(colormap)

# lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")

# colormap = imgviz.label_colormap()
# lbl_pil.putpalette(colormap.flatten())
# lbl_pil.save(save_path)

# if __name__ == '__main__':
#     mask = cv2.imread(r"F:\AAA-projects\ING\3-march\700-tooth\4_split_dataset\Test_Labels\train_2.png", cv2.IMREAD_UNCHANGED)
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
#     colormap = imgviz.label_colormap()
#     print(colormap)
#     lbl_pil.putpalette(colormap.flatten())
#     lbl_pil.save("aaa.png")

# print(img.shape)
# makesave(filename=r"result.png", mask=img)
