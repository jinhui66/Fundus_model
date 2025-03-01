import os
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
from ietk import methods
from ietk import util

def process_fundus_images(input_folder, output_folder=None):
    """
    处理指定文件夹中的所有眼底图像JPG文件：
    1. 读取图像
    2. 裁剪并获取前景掩码
    3. 应用A+B+X增强方法
    4. 应用锐化处理
    5. 保存处理后的图像
    """
    # 如果指定了输出文件夹且不存在，则创建
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有JPG图像
    image_paths = glob(os.path.join(input_folder, "*.jpg"))
    
    for img_path in image_paths:
        # 提取文件名
        img_name = os.path.basename(img_path)
        print(f"正在处理 {img_name}")
        
        # 读取图像 (使用matplotlib确保读取为RGB)
        img = plt.imread(img_path)
        
        # 如果图像范围是0-255，则归一化为0-1
        if img.max() > 1.0:
            img = img / 255.0
        
        # 裁剪图像并获取前景掩码
        I = img.copy()
        I, fg = util.center_crop_and_get_foreground_mask(I)
        
        # 应用A+B+X增强方法
        enhanced_img = methods.brighten_darken(I, 'A+B+X', focus_region=fg)
        
        # 应用锐化处理
        enhanced_img2 = methods.sharpen(enhanced_img, bg=~fg)
        
        # 如果指定了输出文件夹，则保存结果
        if output_folder:
            # 确保值在0-1范围内
            enhanced_img2_save = np.clip(enhanced_img2, 0, 1)
            
            # 保存增强后的图像
            output_path = os.path.join(output_folder, f"{img_name}")
            plt.imsave(output_path, enhanced_img2_save)
            print(f"已保存到 {output_path}")
        
        # # 显示结果
        # f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        # ax1.imshow(img)
        # ax1.set_title("原始图像")
        # ax1.axis('off')
        
        # ax2.imshow(enhanced_img)
        # ax2.set_title("A+B+X增强")
        # ax2.axis('off')
        
        # ax3.imshow(enhanced_img2)
        # ax3.set_title("锐化后")
        # ax3.axis('off')
        
        # f.tight_layout()
        # plt.show()

# 使用示例
if __name__ == "__main__":
    # 替换为您的输入文件夹路径
    input_folder = "/data3/wangchangmiao/jinhui/eye/Training_Dataset"
    # 替换为您的输出文件夹路径（或设置为None以跳过保存）
    output_folder = "/data3/wangchangmiao/jinhui/eye/ietk_Enhanced"
    
    process_fundus_images(input_folder, output_folder)