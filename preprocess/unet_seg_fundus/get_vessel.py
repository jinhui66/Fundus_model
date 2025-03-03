import os
import cv2
import numpy as np
from PIL import Image

def remove_vessels_from_image(original_img_path, mask_img_path, output_path):
    """
    利用血管分割掩码去除原图中的血管部分
    :param original_img_path: 原图路径
    :param mask_img_path: 血管分割掩码路径
    :param output_path: 输出去除血管后的图像路径
    """
    # 读取原图和血管分割掩码
    original_img = cv2.imread(original_img_path, cv2.IMREAD_COLOR)
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    # 将掩码二值化
    _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)

    # 反转掩码
    mask_img_inv = cv2.bitwise_not(mask_img)

    # 去除血管
    result_img = cv2.bitwise_and(original_img, original_img, mask=mask_img_inv)

    # 保存结果
    cv2.imwrite(output_path, result_img)

    # 显示结果
    cv2.imshow('Result Image', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # 输入路径
    original_img_dir = "E:/unet_seg_fundus/data/images"  # 原图文件夹
    mask_img_dir = "./output_mask"  # 血管分割掩码文件夹
    output_dir = "./output_images_without_vessels"  # 输出去除血管后的图像文件夹

    # 检查文件夹是否存在
    assert os.path.exists(original_img_dir), f"Original image directory {original_img_dir} not found."
    assert os.path.exists(mask_img_dir), f"Mask image directory {mask_img_dir} not found."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历文件夹中的所有图像
    for img_filename in os.listdir(original_img_dir):
        original_img_path = os.path.join(original_img_dir, img_filename)
        mask_img_path = os.path.join(mask_img_dir, img_filename)

        # 检查是否为图像文件（这里假设图片扩展名为.tif, .jpg, .png等）
        if img_filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
            if os.path.exists(mask_img_path):  # 确保掩码文件存在
                print(f"Processing image: {original_img_path} with mask: {mask_img_path}")
                output_path = os.path.join(output_dir, img_filename)
                remove_vessels_from_image(original_img_path, mask_img_path, output_path)
            else:
                print(f"Warning: No mask found for {original_img_path}. Skipping this image.")

if __name__ == '__main__':
    main()