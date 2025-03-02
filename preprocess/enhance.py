'''
fundus images preprocess

del_black_or_white  delete borders of fundus images

detect_xyr  using HoughCircles detect circle, if not detected
  suppose the center of the image is the center of the circle.

my_crop_xyz  crot the image based on circle detected

after croped, add some black margin areas,
   so that img aug(random rotate clip) will not delete meaningful edge region
'''

import cv2
import numpy as np
import os
from imgaug import augmenters as iaa

DEL_PADDING_RATIO = 0.02  #used for del_black_or_white
CROP_PADDING_RATIO = 0.02  #used for my_crop_xyr

# del_black_or_white margin
THRETHOLD_LOW = 7
THRETHOLD_HIGH = 180

# HoughCircles
MIN_REDIUS_RATIO = 0.33
MAX_REDIUS_RATIO = 0.6

def del_black_or_white(img1):
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=-1)

    width, height = (img1.shape[1], img1.shape[0])

    (left, bottom) = (0, 0)
    (right, top) = (img1.shape[1], img1.shape[0])

    padding = int(min(width, height) * DEL_PADDING_RATIO)

    for i in range(width):
        array1 = img1[:, i, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            left = i
            break
    left = max(0, left-padding)

    for i in range(width - 1, 0 - 1, -1):
        array1 = img1[:, i, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            right = i
            break
    right = min(width, right + padding)

    for i in range(height):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            bottom = i
            break
    bottom = max(0, bottom - padding)

    for i in range(height - 1, 0 - 1, -1):
        array1 = img1[i, :, :]
        if np.sum(array1) > THRETHOLD_LOW * array1.shape[0] * array1.shape[1] and \
                np.sum(array1) < THRETHOLD_HIGH * array1.shape[0] * array1.shape[1]:
            top = i
            break
    top = min(height, top + padding)

    img2 = img1[bottom:top, left:right, :]

    return img2

def detect_xyr(img_source):
    if isinstance(img_source, str):
        try:
            img = cv2.imread(img_source)
        except:
            raise Exception("image file not found:" + img_source)
        if img is None:
            raise Exception("image file error:" + img_source)
    else:
        img = img_source

    width = img.shape[1]
    height = img.shape[0]

    myMinWidthHeight = min(width, height)

    myMinRadius = round(myMinWidthHeight * MIN_REDIUS_RATIO)
    myMaxRadius = round(myMinWidthHeight * MAX_REDIUS_RATIO)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=450, param1=120, param2=32,
                               minRadius=myMinRadius,
                               maxRadius=myMaxRadius)

    (x, y, r) = (0, 0, 0)
    found_circle = False

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        if (circles is not None) and (len(circles == 1)):
            x1, y1, r1 = circles[0]
            if x1 > (2 / 5 * width) and x1 < (3 / 5 * width) \
                    and y1 > (2 / 5 * height) and y1 < (3 / 5 * height):
                x, y, r = circles[0]
                found_circle = True

    if not found_circle:
        x = img.shape[1] // 2
        y = img.shape[0] // 2
        temp_x = img[int(img.shape[0] / 2), :, :].sum(1)
        r = int((temp_x > temp_x.mean() / 12).sum() / 2)

    return (found_circle, x, y, r)

def my_crop_xyr(img_source, x, y, r, crop_size=None):
    if isinstance(img_source, str):
        try:
            image1 = cv2.imread(img_source)
        except:
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    original_width = image1.shape[1]
    original_height = image1.shape[0]

    (image_height, image_width) = (image1.shape[0], image1.shape[1])

    img_padding = int(min(original_width, original_height) * CROP_PADDING_RATIO)

    image_left = int(max(0, x - r - img_padding))
    image_right = int(min(x + r + img_padding, image_width - 1))
    image_bottom = int(max(0, y - r - img_padding))
    image_top = int(min(y + r + img_padding, image_height - 1))

    if image_width >= image_height:
        if image_height >= 2 * (r + img_padding):
            image1 = image1[image_bottom: image_top, image_left:image_right]
        else:
            image1 = image1[:, image_left:image_right]
    else:
        if image_width >= 2 * (r + img_padding):
            image1 = image1[image_bottom: image_top, image_left:image_right]
        else:
            image1 = image1[image_bottom:image_top, :]

    if crop_size is not None:
        image1 = cv2.resize(image1, (crop_size, crop_size))

    return image1

def add_black_margin(img_source, add_black_pixel_ratio = 0.05):
    if isinstance(img_source, str):
        try:
            image1 = cv2.imread(img_source)
        except:
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    height, width = image1.shape[:2]

    add_black_pixel = int(min(height, width) * add_black_pixel_ratio)

    img_h = np.zeros((add_black_pixel, width, 3))
    img_v = np.zeros((height + add_black_pixel*2, add_black_pixel, 3))

    image1 = np.concatenate((img_h, image1, img_h), axis=0)
    image1 = np.concatenate((img_v, image1, img_v), axis=1)

    return image1

def my_preprocess(img_source, crop_size, train_or_valid='train', img_file_dest=None):
    if isinstance(img_source, str):
        try:
            image1 = cv2.imread(img_source)
        except:
            raise Exception("image file not found:" + img_source)
    else:
        image1 = img_source

    if image1 is None:
        raise Exception("image file error:" + img_source)

    image1 = del_black_or_white(image1)

    min_width_height = min(image1.shape[0], image1.shape[1])

    if min_width_height < 100:
        return None

    image_size_before_hough = crop_size * 2
    if min_width_height > image_size_before_hough:
        crop_ratio = image_size_before_hough / min_width_height
        image1 = cv2.resize(image1, None, fx=crop_ratio, fy=crop_ratio)

    (found_circle, x, y, r) = detect_xyr(image1)

    if train_or_valid == 'train':
        image1 = my_crop_xyr(image1, x, y, r)
        image1 = add_black_margin(image1, add_black_pixel_ratio=0.07)
        image1 = cv2.resize(image1, (crop_size, crop_size))
    else:
        image1 = my_crop_xyr(image1, x, y, r, crop_size)

    if img_file_dest is not None:
        if not os.path.exists(os.path.dirname(img_file_dest)):
            os.makedirs(os.path.dirname(img_file_dest))
        cv2.imwrite(img_file_dest, image1)

    return image1

# def flip_right_images(image, filename):
#     """
#     只对包含'right'的图像进行左右翻转
#     """
#     if 'right' in filename.lower():
#         return cv2.flip(image, 1)  # 1表示水平翻转
#     return None

if __name__ == '__main__':
    input_dir = '/data3/wangchangmiao/jinhui/eye/Training_Dataset'  
    output_dir = '/data3/wangchangmiao/jinhui/eye/Enhanced'
    crop_size = 256

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入文件夹中的所有图像
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            try:
                # 处理原始图像
                img_processed = my_preprocess(
                    input_path, 
                    crop_size=crop_size, 
                    train_or_valid='train'
                )
                
                if img_processed is not None:
                    
                    cv2.imwrite(output_path, img_processed)
                    print(f'Successfully processed: {filename}')
                else:
                    print(f'Failed to process (image too small): {filename}')
                    
            except Exception as e:
                print(f'Error processing {filename}: {str(e)}')

    print('Processing completed!')