import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import cv2

from src import UNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def process_image(model, img_path, roi_mask_path, output_dir, device):
    """
    处理单张图像并进行分割推理，输出去除血管后的眼底图像
    """
    # 加载原始图像
    original_img = Image.open(img_path).convert('RGB')
    original_img_np = np.array(original_img)  # 转换为 NumPy 数组

    # 加载 ROI 掩码
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # 图像预处理：转换为 Tensor 并归一化
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    img = torch.unsqueeze(img, dim=0)  # 扩展 batch 维度

    # 模型推理
    model.eval()  # 进入验证模式
    with torch.no_grad():
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print(f"Inference time for {img_path}: {t_end - t_start:.4f} seconds")

        # 获取预测结果
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)

        # 将前景（血管）对应的像素值设为 255，背景设为 0
        prediction[prediction == 1] = 255
        prediction[prediction == 0] = 0

        # 将不感兴趣的区域（ROI 外部）设为 0
        prediction[roi_img == 0] = 0

        # 使用图像修复技术去除血管
        inpaint_radius = 3  # 修复半径，可以根据需要调整
        result_img = cv2.inpaint(original_img_np, prediction, inpaint_radius, cv2.INPAINT_TELEA)

        # 将结果转换为 PIL 图像
        result_img = Image.fromarray(result_img)

        # 调整图像大小为256x256
        # result_img = result_img.resize((256, 256), Image.Resampling.LANCZOS)

        # 保存结果
        result_path = os.path.join(output_dir, os.path.basename(img_path))
        result_img.save(result_path)
        print(f"Saved result for {img_path} to {result_path}")


def main():
    weights_path = "./save_weights/best_model.pth"
    input_dir = "/data3/wangchangmiao/jinhui/eye/Enhanced"  # 文件夹中包含待推理的图像
    roi_mask_dir = "/data3/wangchangmiao/jinhui/eye/ROI"  # 文件夹中包含掩码图像
    output_dir = "/data3/wangchangmiao/jinhui/eye/image_without_vessel"  # 存放去除血管后的图像

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
        roi_mask_path = os.path.join(roi_mask_dir, img_filename.split('.')[0] + '_mask.jpg')

        # 检查是否为图像文件（这里假设图片扩展名为.tif, .png, .jpg等）
        if img_filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
            if os.path.exists(roi_mask_path):  # 确保掩码文件存在
                print(f"Processing image: {img_path} with mask: {roi_mask_path}")
                process_image(model, img_path, roi_mask_path, output_dir, device)
            else:
                print(f"Warning: No ROI mask found for {img_path}. Skipping this image.")


if __name__ == '__main__':
    main()