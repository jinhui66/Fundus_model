from fundus_crop import fundus_crop
from fundus_enhance1 import fundus_enhance1

if __name__ == '__main__':
    input_dir = r'/data3/wangchangmiao/jinhui/eye/Origin_Dataset'  
    middle_dir = r'/data3/wangchangmiao/jinhui/eye/Temp_enhanced'  
    output_dir = '/data3/wangchangmiao/jinhui/eye/Enhanced'
    
    fundus_enhance1(input_dir, middle_dir)
    fundus_crop(middle_dir, output_dir)