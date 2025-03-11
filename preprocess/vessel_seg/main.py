from predict_mask import predict_mask
from predict import predict
from rest import rest
from ROI import ROI

if __name__ == '__main__':
    dir_prefix = "/data3/wangchangmiao/jinhui/eye/ietk_seg/"
    input_dir = "/data3/wangchangmiao/jinhui/eye/ietk_Enhanced"  # 替换为你的图像文件夹路径
    ROI_dir = dir_prefix+"ROI" # 替换为你想保存掩码的文件夹路径
    vessel_enhanced_dir = dir_prefix+"vessel_enhanced"
    vessel_mask_dir = dir_prefix+"vessel_mask" 
    image_without_vessel_dir = dir_prefix+"image_without_vessel"
    
    ROI(input_dir, ROI_dir)
    predict(input_dir, vessel_enhanced_dir)
    predict_mask(input_dir, ROI_dir, vessel_mask_dir)
    rest(input_dir, ROI_dir, image_without_vessel_dir)
    
    
    