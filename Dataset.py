import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer
from RET_CLIP.clip import tokenize

class DoubleEyesDataset(Dataset):
    def __init__(self, csv_file, img_prefix, if_text=False, transform=None):
        """
        初始化 Dataset
        :param csv_file: str, CSV 文件路径
        :param img_prefix: str, 图片路径前缀
        :param transform: torchvision.transforms, 数据增强和预处理
        """
        self.data = pd.read_csv(csv_file)
        self.img_prefix = img_prefix
        self.labels = {'N': 0, 'D': 1, 'G': 2, 'C': 3, 'A': 4, 'H': 5, 'M': 6, 'O': 7}
        self.if_text = if_text
        # self.text = {'Patient Age': 0, 'Patient Sex': 1}
        self.transform = transform

    def __len__(self):
        """
        返回数据集的长度
        """
        return len(self.data)

    def get_label_index(self, row):
        """
        根据标签列返回类别索引
        :param row: DataFrame 的一行
        :return: int, 类别索引
        """
        label_index = torch.zeros(8)
        for label, index in self.labels.items():
            if row[label] == 1:
                label_index[index] = 1

        return label_index

    def get_text(self, row):
        text = f"左眼：{row['Left-Feature']} 右眼: {row['Right-Feature']}"
        # text = torch.zeros(2)
        # text[0] = row['Patient Age']
        # text[1] = 1 if row['Patient Sex']=='Male' else -1
        # print(text)
        text = tokenize(text, context_length=50)[0]
        # print(text,text.shape)
        # text = torch.tensor([text])
        return text

    def get_table(self, row):
        table = torch.tensor([0, 0])
        # print(row['Patient Age'], row['Patient Sex'])
        table[0] = row['Patient Age']
        table[1] = 1 if row['Patient Sex'] == 'Male' else -1
        # print(table)
        return table

    def __getitem__(self, idx):
        """
        获取指定索引的数据
        :param idx: int, 数据索引
        :return: tuple (image, label_index)
        """
        idx = int(idx)
        if isinstance(idx, int):
            row = self.data.iloc[idx]
        else:
            raise TypeError("Index must be an integer.")

        # 构造图片路径
        # print(row)
        left_img_name = row.iloc[3]  
        left_img_path = f"{self.img_prefix}/{left_img_name}"
        
        right_img_name = row.iloc[4]  
        right_img_path = f"{self.img_prefix}/{right_img_name}"

        # 打开图片
        left_image = Image.open(left_img_path).convert('RGB')
        right_image = Image.open(right_img_path).convert('RGB')

        # 数据增强和预处理
        if self.transform:
            image = self.transform(image)

        # 获取类别索引
        label_index = self.get_label_index(row)
        table = self.get_table(row)
        
        if self.if_text == True:
            text = self.get_text(row)
            
            # print(text)
            # print(text)
            return left_image, right_image, label_index, table, text
        else:
            return left_image, right_image, label_index, table
        
    # def text_to_tensor(text):
    #     tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    #     text_tensor = tokenizer(text, padding='max_length',
    #             max_length=10,
    #             truncation=True,
    #             return_tensors="pt")
    #     return text_tensor
    
class DoubleTransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y, *rest = self.subset[index]
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y, *rest

    def __len__(self):
        return len(self.subset)

