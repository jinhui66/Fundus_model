import io
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import transforms
from src import UNet
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI()

# 全局变量存储模型和设备
device = None
model = None

def load_model():
    """
    加载模型和初始化全局变量
    """
    global device, model
    
    weights_path = "./save_weights/best_model.pth"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型
    model = UNet(in_channels=3, num_classes=2, base_c=32)
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.to(device)
    model.eval()

def process_image(image: Image.Image):
    """
    处理图像的核心函数
    """
    # 图像预处理
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    img = data_transform(image)
    img = torch.unsqueeze(img, dim=0)

    with torch.no_grad():
        output = model(img.to(device))
        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        
        # 处理原图检测黑色区域
        original_array = np.array(image)
        black_threshold = 30
        black_regions = np.all(original_array < black_threshold, axis=2)
        
        # 创建RGBA掩码
        mask_rgba = np.zeros((prediction.shape[0], prediction.shape[1], 4), dtype=np.uint8)
        mask_condition = (prediction == 1) & (~black_regions)
        mask_rgba[mask_condition] = [255, 255, 0, 128]
        mask = Image.fromarray(mask_rgba, mode='RGBA')
        
        # 合成结果图
        image = image.convert('RGBA')
        result_img = Image.alpha_composite(image, mask)
        result_img = result_img.convert('RGB')
        
        return result_img

@app.on_event("startup")
async def startup_event():
    """
    服务启动时加载模型
    """
    load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    处理上传的图像并返回预测结果
    """
    try:
        # 读取上传的图像
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # 处理图像
        result_img = process_image(image)
        
        # 保存结果到临时文件
        output_path = f"temp_{file.filename}"
        result_img.save(output_path)
        
        # 返回处理后的图像
        return FileResponse(output_path)
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 