# 数据集转换
# 将原先得255数据集转换为新得01数据集，保证代码测试
import copy
import os
import os.path as osp
import cv2


def data_convert(src_folder, target_folder):
    if osp.isdir(target_folder) == False:
        os.mkdir(target_folder)
    image_names = os.listdir(src_folder)
    for image_name in image_names:
        image_path = osp.join(src_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(img)
        img_g = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # 应该找到对应的灰度图的值。
        img_1 = img == [255, 0, 0]
        img_1 = img_1[:, :, 0] & img_1[:, :, 1] & img_1[:, :, 2]
        img_g[img_1] = 1

        img_2 = img == [255, 0, 255]
        img_2 = img_2[:, :, 0] & img_2[:, :, 1] & img_2[:, :, 2]
        img_g[img_2] = 2

        img_3 = img == [0, 139, 0]
        img_3 = img_3[:, :, 0] & img_3[:, :, 1] & img_3[:, :, 2]
        img_g[img_3] = 3

        img_4 = img == [255, 127, 80]
        img_4 = img_4[:, :, 0] & img_4[:, :, 1] & img_4[:, :, 2]
        img_g[img_4] = 4

        img_5 = img == [255, 255, 0]
        img_5 = img_5[:, :, 0] & img_5[:, :, 1] & img_5[:, :, 2]
        img_g[img_5] = 5

        img_6 = img == [0, 255, 255]
        img_6 = img_6[:, :, 0] & img_6[:, :, 1] & img_6[:, :, 2]
        img_g[img_6] = 6

        save_path = osp.join(target_folder, image_name)
        cv2.imwrite(save_path, img_g)


if __name__ == '__main__':
    data_convert(src_folder="G:/money/project_dz/5_May/700_remote_seg/Split_dataset/Training_Labels",
                 target_folder="G:/money/project_dz/5_May/700_remote_seg/Split_dataset/Training_Labels_6")
