import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import copy
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from Config import parse_args
from timm.models.layers import DropPath, trunc_normal_
from datetime import datetime
from pathlib import Path
from typing import Sequence
from functools import partial, reduce
import torchvision
import time
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from Dataset import DoubleEyesDataset, DoubleTransformedSubset
from utils.metrics import Metric_Manager_Normal, Metric_Manager
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import json
from RET_CLIP.clip.model import CLIP
from RET_CLIP.get_model import get_retclip_model
from model.CLIP_Head import MoeLayer, MoeArgs
import torch.nn.functional as F

class Fusion_Head(nn.Module):
    def __init__(self, num_classes=7, activation="sigmoid"):
        super(Fusion_Head, self).__init__()
        self.num_classes = num_classes
        if activation == "tanh":
            self.head = nn.Sequential(nn.Linear(1024 + 2, 128), nn.Tanh(), nn.Linear(128, num_classes))
        elif activation == "sigmoid":
            self.head = nn.Sequential(nn.Linear(1024 + 2, 128), nn.Sigmoid(), nn.Linear(128, num_classes))
    
    def forward(self, vision_feature, label):
        fusion = torch.cat((vision_feature, label), dim=1)
        output = self.head(fusion)
        return output

class Layer(nn.Module):
    def __init__(self, linear, num_classes=7):
        super(Layer, self).__init__()
        self.num_classes = num_classes
        self.linear = linear
    def forward(self, vision_feature, *label):
        x = self.linear(vision_feature)
        return x

def to0_1(tensor):
    bool_tensor = tensor > 0.5
    # 将布尔型张量转换为浮点型张量，True变为1.0，False变为0.0
    result_tensor = bool_tensor.float()
    return result_tensor

def k_fold_cross_validation(device, eyes_dataset, epochs, k_fold=5,
                            batch_size=8, workers=2, print_freq=1, checkpoint_dir="./result",
                            best_result_model_path="model", total_transform=None):
    start = time.time()
    # skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    # best_fold_metrics = []
    best_acc_overall = 0.
    epoch_acc_dict = {}
    
    train_metric = Metric_Manager_Normal(num_classes=8)
    valid_metric = Metric_Manager_Normal(num_classes=8)
    
    # 获取数据集的标签
    # labels = [data[1] for data in eyes_dataset]  # 假设dataset[i]的第2项是label
    kf = KFold(n_splits=5, shuffle=True, random_state=0)  # init KFold
    # Iterations = 1
    
    for fold, (train_index, test_index) in enumerate(kf.split(eyes_dataset)):  # split
    # for fold, (train_idx, val_idx) in enumerate(skf.split(eyes_dataset, labels), 1):
        # get train, val
        k_train_fold = Subset(eyes_dataset, train_index)
        k_test_fold = Subset(eyes_dataset, test_index)
        # 应用转换
        train_dataset = DoubleTransformedSubset(k_train_fold, transform=total_transform['train_transforms'])
        val_dataset = DoubleTransformedSubset(k_test_fold, transform=total_transform['validation_transforms'])

        # package type of DataLoader
        train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        eval_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
        
        X = torch.zeros(7)
        Count = 0
        for (_, _, labels, *_) in train_dataloader:
            X += torch.sum(labels[:, 1:],dim=0)
            Count += labels.shape[0]
            # break
            # print(X, Count)
        # X = torch.tensor([100., 100., 100., 100., 100., 100., 100.])
        print("各疾病总数:", X, Count)
        X = Count / X - 1
        print("各疾病损失权重:", X)
        # X = torch.tensor([ 4.7506, 12.1771, 12.9758, 16.4697, 32.9118, 15.4714,  2.0624])
        # X = torch.tensor([ 4.5566, 12.3295, 11.8827, 16.4697, 37.4333, 15.4714,  1.9909])
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=X).to(device)  # 损失函数

        # model = DoubleImageModel7Class(num_classes=7, pretrained_path=args.pretrainedModelPath, if_text=True).to(device)
        model = get_retclip_model()
        model = model.to(device)
        model.eval()
        # head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 7), nn.Sigmoid())
        # single_expert = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 7))
        expert1 = Layer(nn.Sequential(nn.Linear(1024, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 7)))
        expert2 = Layer(nn.Sequential(nn.Linear(1024, 64), nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, 7)))
        expert3 = Fusion_Head(activation="tanh")
        expert4 = Fusion_Head(activation="sigmoid")
        expert5 = Layer(nn.Sequential(nn.Linear(1024, 7)))

        experts = [expert1, expert2, expert3, expert4, expert5]
        gate = nn.Sequential(nn.Linear(1024, 5))
        head = MoeLayer(experts, gate, MoeArgs())
        head = head.to(device)
        params = [p for p in head.parameters() if p.requires_grad]
        # print(params)
        optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-7, betas=(0.9, 0.98))
        total_params = sum(p.numel() for p in params)
        print('总参数个数:{}'.format(total_params))

        best_score = 0.
        best_epoch = 0
        best_acc = 0.
        best_metrics = {}
        for e in range(1, epochs + 1):
            train_metric.reset()
            valid_metric.reset()
            
            # model.train()
            head.train()
            total_train_loss = 0.0
            total_eval_loss = 0.0
            train_iterator = tqdm(train_dataloader, desc=f"Training Epoch {e}", unit="batch")
            for batch_idx, (left_images, right_images, labels, table, text) in enumerate(train_iterator):
                left_images, right_images, labels = left_images.to(device), right_images.to(device), labels.to(device)

                # print(text,text.shape)
                table, text = table.to(device), text.to(device)  
                
                optimizer.zero_grad()
                
                image_features, text_feature, *_ = model(left_images, right_images, text=text)
                # feature 
                # print(image_features.shape, text_feature.shape, scale.shape)
                # print(vision_feature[0])
                feature = torch.cat((image_features, text_feature), dim=-1)
                predictions = head(feature, table)

                # templabel = labels[:, 1:].clone()  # 重要：使用 .copy() 防止修改原数组
                # templabel[templabel == 1] = 0.95  # 替换 1 为 0.95
                # templabel[templabel == 0] = 0.05  # 替换 0 为 0.05
                # print(templabel)
                loss = loss_fn(predictions, labels[:, 1:])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item()
                predictions = F.sigmoid(predictions)
                # print(predictions)
                predictions = to0_1(predictions)
                

                train_metric.update(predictions, labels)
                # print(train_metric.get_metrics())

            train_accuracy, train_recall, train_precision, train_specificity = train_metric.get_metrics()
            TP, TN, FP, FN = train_metric.get_matrix()
            score = train_metric.compute_score()
            # total_score = score[0]
            total_train_loss /= len(train_index)

            print(f"Epoch [{e}/{epochs}]: train_loss={total_train_loss:.3f}, accuracy={train_accuracy}, recall={train_recall}, precision={train_precision}, specificity={train_specificity}, train_score={score}")
            if e % print_freq == 0:
                output_result = f"Epoch [{e}/{epochs}]: \n train_loss={total_train_loss:.3f}, \n accuracy={train_accuracy}, \n recall={train_recall}, \n precision={train_precision}, \n specificity={train_specificity}, \n train_score:{score} \n 矩阵: \nTP:{TP}\nTN:{TN}\nFP:{FP}\nFN:{FN} \n"

                # 将完整的字符串写入文件
                with open(otuput_file, 'a') as f:
                    f.write(output_result + '\n')  # 在最后添加一个换行符以保持格式整洁
            
            with torch.no_grad():
                # model.eval()
                head.eval()
                
                eval_iterator = tqdm(eval_dataloader, desc=f"Evaluating Epoch {e}", unit="batch")
                for batch_idx, (left_images, right_images, labels, table, text) in enumerate(eval_iterator):
                    left_images, right_images, labels = left_images.to(device), right_images.to(device), labels.to(device)
                    # print(text.shape)
                    table, text = table.to(device), text.to(device)  
                    image_features, text_feature, *_ = model(left_images, right_images, text=text)
                    # feature 
                    # print(image_features.shape, text_feature.shape, scale.shape)
                    # print(vision_feature[0])
                    feature = torch.cat((image_features, text_feature), dim=-1)
                    predictions = head(feature, table)
                    
                    # templabel = labels[:, 1:].clone() 
                    # templabel[templabel == 1] = 0.95  # 替换 1 为 0.95
                    # templabel[templabel == 0] = 0.05  # 替换 0 为 0.05

                    loss = loss_fn(predictions, labels[:, 1:])
                    total_eval_loss += loss.item()
                    
                    # print(predictions[:, 0], labels[:, 0])
                    predictions = F.sigmoid(predictions)
                    # print(predictions)
                    predictions = to0_1(predictions)
                    
                    # _, val_predicted = torch.max(predictions.data, 1)

                    valid_metric.update(predictions, labels)
                    
                valid_accuracy, valid_recall, valid_precision, valid_specificity = valid_metric.get_metrics()
                score = valid_metric.compute_score()
                TP, TN, FP, FN = valid_metric.get_matrix()
                total_score = score[0]

                total_eval_loss /= len(test_index)

            if total_score > best_score:
                best_score = total_score
                best_epoch = e
                torch.save(head.state_dict(), f'./checkpoint/head-final_{fold}.pth')

                
            print(f"Epoch [{e}/{epochs}]: val_loss={total_eval_loss:.3f}, accuracy={valid_accuracy}, recall={valid_recall}, precision={valid_precision}, specificity={valid_specificity}, val_score={score}")
            
            if e % print_freq == 0:
                output_result = f"Epoch [{e}/{epochs}]: \n val_loss={total_eval_loss:.3f}, \n accuracy={valid_accuracy}, \n recall={valid_recall}, \n precision={valid_precision}, \n specificity={valid_specificity}, \n val_score={score} \n 矩阵: \nTP:{TP}\nTN:{TN}\nFP:{FP}\nFN:{FN} \n"

                # 将完整的字符串写入文件
                with open(otuput_file, 'a') as f:
                    f.write(output_result + '\n')  # 在最后添加一个换行符以保持格式整洁



        # best_fold_metrics.append(best_metrics)
        print(f"Fold {fold} Best Epoch: {best_epoch}, Best Score: {best_score:.3f}")
        with open(otuput_file, 'a') as f:
            f.write(f"Fold {fold} Best Epoch: {best_epoch}, Best Score: {best_score:.3f} \n")  # 在最后添加一个换行符以保持格式整洁
        epoch_acc_dict[fold] = best_acc

        # if best_acc > best_acc_overall:
        #     best_acc_overall = best_acc
        # break

    end = time.time()
    print(f"Cross-validation result:{epoch_acc_dict}")
    print(f"Total training time: {(end - start) // 60}m {(end - start) % 60}s")
    



if __name__ == '__main__':
    current_time = "{0:%Y%m%d_%H_%M}".format(datetime.now())
    args = parse_args()
    # 获取运行的设备
    if args.device != 'cpu':
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Define transforms for each dataset separately
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    
    image_size = 224

    train_validation_test_transform={
        'train_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        # transforms.RandomHorizontalFlip(0.5),
        # transforms.RandomRotation(45),
        # transforms.RandomAdjustSharpness(1.3, 1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
        'validation_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ]),
        'test_transforms':transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
        ])
    }
    
    data_dir = "/data3/wangchangmiao/jinhui/DATA/fundus/Enhanced"
    # 初始化自定义数据集
    dataset = DoubleEyesDataset(csv_file="./data/double_valid_data.csv",
                            img_prefix=data_dir,
                            transform=None,
                            if_text=True)

    # 定义模型保存的文件夹
    model_dir = args.checkpoint_dir
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    # 训练的总轮数
    EPOCH = 500
    epoch = 0
    otuput_file = "Final.txt"
    k_fold_cross_validation(device, dataset, EPOCH, k_fold=args.k_split_value,
                            batch_size=args.batch_size, workers=2, print_freq=1, checkpoint_dir=model_dir,
                            best_result_model_path="model", total_transform=train_validation_test_transform)