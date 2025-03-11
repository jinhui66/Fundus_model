import cv2
import numpy as np
import os

def enhance_fundus_image(image):
    """
    使用CLAHE算法增强眼底图像
    Args:
        image: 输入的BGR格式图像
    Returns:
        enhanced_image: 增强后的图像
    """
    # 转换到LAB色彩空间
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    
    # 对L通道进行CLAHE处理
    l_clahe = clahe.apply(l)
    
    # 合并通道
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # 转换回BGR色彩空间
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return enhanced_image

def process_fundus_images(input_dir, output_dir):
    """
    处理文件夹中的所有眼底图像
    Args:
        input_dir: 输入图像文件夹路径
        output_dir: 输出图像文件夹路径
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 读取图像
                img_array = np.fromfile(os.path.join(input_dir, filename), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is None:
                    print(f'Failed to read image: {filename}')
                    continue
                
                # 增强图像
                enhanced_img = enhance_fundus_image(img)
                
                # 保存增强后的图像
                output_path = os.path.join(output_dir, filename)
                _, img_encoded = cv2.imencode('.jpg', enhanced_img)
                img_encoded.tofile(output_path)
                
                print(f'Successfully enhanced: {filename}')
                
            except Exception as e:
                print(f'Error processing {filename}: {str(e)}')

def fundus_enhance1(input_dir, output_dir):
    # 设置输入输出路径

    
    # 处理图像
    process_fundus_images(input_dir, output_dir)
    print('Enhancement completed!')