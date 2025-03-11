import cv2
import numpy as np
import os

def auto_generate_roi_mask(image_path, output_mask_path, method='threshold'):
    """
    自动生成 ROI 掩码
    :param image_path: 输入图像路径
    :param output_mask_path: 输出掩码路径
    :param method: 生成掩码的方法，可选 'threshold' 或 'edge'
    """
    # 读取图像并转换为灰度
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # 方法 1: 基于阈值分割
    if method == 'threshold':
        # 使用 Otsu 算法自动计算阈值
        _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 方法 2: 基于边缘检测
    elif method == 'edge':
        # 使用 Canny 边缘检测
        edges = cv2.Canny(image, 100, 200)

        # 查找轮廓并填充
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    else:
        print(f"Error: Unknown method '{method}'. Supported methods are 'threshold' and 'edge'.")
        return

    # 保存掩码图像
    cv2.imwrite(output_mask_path, mask)
    print(f"ROI mask saved to {output_mask_path}")


def batch_process_images(input_dir, output_dir, method='threshold'):
    """
    批量处理文件夹中的图像
    :param input_dir: 输入图像文件夹路径
    :param output_dir: 输出掩码文件夹路径
    :param method: 生成掩码的方法，可选 'threshold' 或 'edge'
    """
    # 检查输入文件夹是否存在
    if not os.path.exists(input_dir):
        print(f"Error: Input directory {input_dir} does not exist.")
        return

    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 遍历输入文件夹中的所有图像文件
    for filename in os.listdir(input_dir):
        # 检查文件是否为图像（支持常见格式）
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp', '.gif')):
            # 输入图像路径
            image_path = os.path.join(input_dir, filename)
            # 输出掩码路径（与输入图像同名，保存到输出文件夹）
            output_mask_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_mask.jpg")

            # 自动生成 ROI 掩码
            print(f"Processing image: {image_path}")
            auto_generate_roi_mask(image_path, output_mask_path, method)


# 示例用法
def ROI(input_dir, output_dir):
    # # 输入图像文件夹路径
    # input_dir = "/data3/wangchangmiao/jinhui/eye/Enhanced"  # 替换为你的图像文件夹路径
    # # 输出掩码文件夹路径
    # output_dir = "/data3/wangchangmiao/jinhui/eye/ROI" # 替换为你想保存掩码的文件夹路径

    # 选择生成掩码的方法：'threshold' 或 'edge'
    method = 'threshold'  # 或者 'edge'

    # 批量处理图像
    batch_process_images(input_dir, output_dir, method)