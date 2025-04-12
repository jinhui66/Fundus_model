import torch
import torch.nn as nn
from torchvision import transforms
from model import *
from PIL import Image
from RET_CLIP.get_model import get_retclip_model
from torch.nn.functional import sigmoid
import pandas as pd
import os

def to0_1(tensor):
    # 阈值
    threshold = 0.5
    bool_tensor = tensor > threshold

    # 将布尔型张量转换为浮点型张量，True变为1.0，False变为0.0
    result_tensor = bool_tensor.float()
    return result_tensor

# 读取和转换图像
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)

if __name__ == '__main__':
    illness_names = ['正常', '糖尿病', '青光眼', '白内障', 'AMD', '高血压', '近视', '其他疾病']

    # 预训练模型路径
    pretrainedModelPath = r"./checkpoint/head2-_0.pth"
    
    # 定义模型
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    model = get_retclip_model()
    model = model.to(device)
    head = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 7), nn.Sigmoid())
    pretrained_dict = torch.load(pretrainedModelPath)
    head.load_state_dict(pretrained_dict)
    head = head.to(device)
    head.eval()

    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    prefix_dir = '/data3/wangchangmiao/jinhui/DATA/fundus_test/Enhanced'
    csv_file = '/data3/wangchangmiao/jinhui/haha/data/validation_.csv'
    output_csv_file = '/data3/wangchangmiao/jinhui/haha/data/predictions.csv'

    data = pd.read_csv(csv_file)
    image_size = 224

    # 定义一个转换，将PIL Image转换为Tensor
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)  # 标准化
    ])

    results = []  # 用于存储预测结果

    for index, row in data.iterrows():
        left_image_name = row.iloc[3]  
        left_image_path = f"{prefix_dir}/{left_image_name}"
        right_image_name = row.iloc[4]  
        right_image_path = f"{prefix_dir}/{right_image_name}"
        # 读取图片
        left_image = load_image(left_image_path).to(device)
        right_image = load_image(right_image_path).to(device)

        vision_feature, *_ = model(left_image, right_image, text=None)
        predictions = head(vision_feature)
        predictions = predictions.squeeze(0)
        predictions = to0_1(predictions)

        if torch.all(predictions == 0):  # 检查是否全为0
            predictions = torch.cat([torch.tensor([1.0]).to(device), predictions], dim=0)
        else:
            predictions = torch.cat([torch.tensor([0.0]).to(device), predictions], dim=0)

        # 将预测结果转换为整数并存储
        predictions_int = predictions.int().cpu().numpy().tolist()  # 转换为CPU上的numpy数组并转为列表
        results.append({
            'image_left': left_image_name,
            'image_right': right_image_name,
            'predictions': predictions_int
        })
        print(results)

    # 将结果写入CSV文件
    output_data = []
    for result in results:
        row_data = {
            'image_left': result['image_left'],
            'image_right': result['image_right']
        }
        row_data.update({illness_names[i]: result['predictions'][i] for i in range(len(illness_names))})
        output_data.append(row_data)

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(output_csv_file, index=False)

    print(f"预测结果已保存到 {output_csv_file}")