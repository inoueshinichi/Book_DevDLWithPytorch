# パッケージのimport
import glob
import os.path as osp
import random
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False


# 乱数シード (共通)
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# 前処理クラス
class ImageTransform():

    def __init__(self, resize, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),  # ランダムクロップ
                transforms.RandomHorizontalFlip(),                       # ランダムフリップ(水平)
                transforms.ToTensor(),                                   # Tensor変換
                transforms.Normalize(mean, std)                          # 標準化
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize),                               # リサイズ
                transforms.CenterCrop(resize),                           # クロップ
                transforms.ToTensor(),                                   # Tensor変換
                transforms.Normalize(mean, std)                                   # 標準化
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


# アリとハチの画像データセットを作成
class HymenopteraDataset(data.Dataset):

    def __init__(self, file_list, transform=None, phase='train'):
        self.file_list = file_list
        self.transform = transform
        self.phase = phase


    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, index):

        img_path = self.file_list[index]
        img = Image.open(img_path)

        img_transformed = self.transform(img, self.phase) # torch.Size([3, 224, 224])

        if self.phase == 'train':
            label = img_path[30:34]
        elif self.phase == 'val':
            label = img_path[28:32]

        if label == 'ants':
            label = 0
        elif label == 'bees':
            label = 1

        return img_transformed, label



# アリとハチの画像へのファイルパスのリストを作成する
def make_datapath_list(phase='train'):

    rootpath = "./data/hymenoptera_data/"
    target_path = osp.join(rootpath + phase + '/**/*.jpg')
    print(target_path)

    # ファイルパスを取得
    path_list = []
    for path in glob.glob(target_path):
        path_list.append(path)

    for i, path in enumerate(path_list):
        print(str(i) + ": " +  path)

    return path_list


# データセット作成
def makeDataset():

    # 訓練
    train_list = make_datapath_list(phase='train')

    # 検証
    val_list = make_datapath_list(phase='val')

    # データセット
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)
    train_dataset = HymenopteraDataset(file_list=train_list, transform=transform, phase='train')
    val_dataset = HymenopteraDataset(file_list=val_list, transform=transform, phase='val')

    # 動作確認
    index = 0
    print(train_dataset.__getitem__(index)[0].size()) # img_transformed
    print(train_dataset.__getitem__(index)[1])        # label

    print("Training Data: ")
    for train_index in range(0, train_dataset.__len__()):
        print("train" + "_" + str(train_index) +  " :" + str(train_dataset.__getitem__(train_index)[1]))

    return train_dataset, val_dataset


def dataLoader(train_dataset = None, val_dataset = None):
    
    # ミニバッチサイズ
    batch_size  = 32

    # DataLoader作成
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle = True
    )

    # 辞書にまとめる
    datasetloaders_dict = { 'train': train_dataloader, 'val': val_dataloader }

    # 動作確認
    batch_iterator = iter(datasetloaders_dict['train']) # イテレータに変換
    inputs, labels = next(batch_iterator) # 1番目の要素を取り出す
    print(inputs.size())
    print(labels)

    return datasetloaders_dict



def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    
    # epochのループ
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('--------')

        # epochごとの学習と検証ループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train() # 訓練モードのモデル
            else:
                net.eval()  # 検証モードのモデル

            epoch_loss = 0.0   # epochの損失和
            epoch_corrects = 0 # epochの正解数 

            # 未学習時の検証性能を確かめるため、epoch=0の訓練では未学習のモデルを使用する
            if (epoch == 0) and (phase == 'train'):
                continue

            # データローダからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                # optimzerを初期化
                optimizer.zero_grad()

                # forward計算
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)              # 出力
                    loss = criterion(outputs, labels)  # 誤差
                    _, preds = torch.max(outputs, 1)   # 予測ラベル

                    # 訓練時はbackpropagation
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    # イテレーションの計算結果
                    # lossの合計を更新
                    epoch_loss += loss.item() * inputs.size(0) # 平均loss * バッチ数
                    # 正解数の合計を更新
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率を表示
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

