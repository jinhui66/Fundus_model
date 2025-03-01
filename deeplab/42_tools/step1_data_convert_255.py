import os
import os.path as osp
import cv2


# 颜色变化
def data_convert(src_folder, target_folder):
    if osp.isdir(target_folder) == False:
        os.mkdir(target_folder)
    image_names = os.listdir(src_folder)
    for image_name in image_names:
        image_path = osp.join(src_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img[img > 0] = 1
        # img[img==255] = 2
        save_path = osp.join(target_folder, image_name)
        cv2.imwrite(save_path, img)


if __name__ == '__main__':
    data_convert(src_folder=r"F:\AAA-projects\ING\500-mmseg\kaggle_split_data\Training_Labels_src",
                 target_folder=r"F:\AAA-projects\ING\500-mmseg\kaggle_split_data\Training_Labels")
