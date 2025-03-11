import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def process_image(model, img_path, output_dir, device):
    """
    处理单张图像并进行分割推理
    """
    # load image
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print(f"Inference time for {img_path}: {t_end - t_start:.4f} seconds")

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        
        # 将原图转换为numpy数组以检测黑色区域
        original_array = np.array(original_img)
        # 定义黑色区域的阈值（可以根据需要调整）
        black_threshold = 1
        # 检测黑色区域（R,G,B都小于阈值的像素）
        black_regions = np.all(original_array < black_threshold, axis=2)
        
        # 创建RGBA格式的掩码图像
        mask_rgba = np.zeros((prediction.shape[0], prediction.shape[1], 4), dtype=np.uint8)
        # 设置黄色 (RGB: 255, 255, 0)，alpha=128表示50%透明度
        # 只在非黑色区域的前景位置设置掩码
        mask_condition = (prediction == 1) & (~black_regions)
        mask_rgba[mask_condition] = [255, 255, 0, 128]
        mask = Image.fromarray(mask_rgba, mode='RGBA')
        
        # 将原图转换为RGBA模式
        original_img = original_img.convert('RGBA')
        
        # 将掩码覆盖到原图上
        result_img = Image.alpha_composite(original_img, mask)
        
        # 调整大小为256x256
        # result_img = result_img.resize((256, 256), Image.BILINEAR)
        
        # 保存结果
        result_path = os.path.join(output_dir, os.path.basename(img_path))
        # 如果需要保存为JPG格式，需要先转换回RGB模式
        result_img = result_img.convert('RGB')
        result_img.save(result_path)
        print(f"Saved result for {img_path} to {result_path}")


def predict(input_dir, output_dir):
    weights_path = "./save_weights/best_model.pth"

    # input_dir = r"/data3/wangchangmiao/jinhui/eye/Enhanced"   # 文件夹中包含待推理的图像
    # #input_dir = "./images"
    # output_dir = "/data3/wangchangmiao/jinhui/eye/vessel_enhanced"  # 存放推理结果

    # 检查文件夹和模型文件是否存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_dir), f"Input directory {input_dir} not found."
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))

    # 创建模型并加载权重
    model = UNet(in_channels=3, num_classes=2, base_c=32)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)

    # 遍历文件夹中的所有图像
    for img_filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_filename)

        # 检查是否为图像文件（这里假设图片扩展名为.tif, .jpg, .png等）
        if img_filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
            print(f"Processing image: {img_path}")
            process_image(model, img_path, output_dir, device)
