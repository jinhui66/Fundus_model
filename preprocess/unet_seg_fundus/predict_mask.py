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


def process_image(model, img_path, roi_mask_path, output_dir, device):
    """
    处理单张图像并进行分割推理
    """
    # load image
    original_img = Image.open(img_path).convert('RGB')

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

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
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 将不感兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0

        mask = Image.fromarray(prediction)

        # 保存结果
        result_path = os.path.join(output_dir, os.path.basename(img_path))
        mask.save(result_path)
        print(f"Saved result for {img_path} to {result_path}")


def main():
    weights_path = "./save_weights/best_model.pth"
    input_dir = "./images"  # 文件夹中包含待推理的图像
    roi_mask_dir = "./mask"  # 文件夹中包含掩码图像
    output_dir = "./output_mask"  # 存放推理结果

    # 检查文件夹和模型文件是否存在
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(input_dir), f"Input directory {input_dir} not found."
    assert os.path.exists(roi_mask_dir), f"ROI mask directory {roi_mask_dir} not found."
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
        roi_mask_path = os.path.join(roi_mask_dir, img_filename.split('.')[0] + '_mask.gif')

        # 检查是否为图像文件（这里假设图片扩展名为.tif, .jpg, .png等）
        if img_filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
            if os.path.exists(roi_mask_path):  # 确保掩码文件存在
                print(f"Processing image: {img_path} with mask: {roi_mask_path}")
                process_image(model, img_path, roi_mask_path, output_dir, device)
            else:
                print(f"Warning: No ROI mask found for {img_path}. Skipping this image.")


if __name__ == '__main__':
    main()
