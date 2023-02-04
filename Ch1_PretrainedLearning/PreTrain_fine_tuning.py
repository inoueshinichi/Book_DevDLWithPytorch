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

import sys
sys.path.append("./utils/")

from utils.dataloader_image_classification import ImageTransform, make_datapath_list, HymenopteraDataset, makeDataset, dataLoader

"""ベタにファインチューニングプログラムを書く"""

# データセット
train_dataset, val_dataset = makeDataset()

# データローダー
dataloaders_dict = dataLoader(train_dataset=train_dataset, val_dataset=val_dataset)

# ネットワークモデル作成
use_pretained = True
net = models.vgg16(pretrained=use_pretained) # 学習済みモデルVGG-16をロード
net.classifier[6] = nn.Linear(in_features=4096, out_features=2) # 出力層の出力ユニットをアリとハチの2つに付け替える
net.train() # 訓練モードに設定
print('ネットワーク設定完了: 学習済み重みを読み込み、訓練モードに設定しました。')

 # 損失関数の設定
criterion = nn.CrossEntropyLoss()


"""ファインチューニング"""
#################
# 最適化手法の設定 #
#################

# 異なる学習率毎にlistを作成
params_to_update_1 = []
params_to_update_2 = []
params_to_update_3 = []

# 学習させる層のパラメータ名を指定
update_param_names_1 = ["features"]
update_param_names_2 = ["classifier.0.weight", "classifier.0.bias", "classifier.3.weight", "classifier.3.bias"]
update_param_names_3 = ["classifier.6.weight", "classifier.6.bias"]

# パラメータ毎に各リストに格納する
for name, param, in net.named_parameters():
    if update_param_names_1[0] in name:
        param.requires_grad = True
        params_to_update_1.append(param)
        print("params_to_update_1に格納: ", name)
    elif name in update_param_names_2:
        param.requires_grad = True
        params_to_update_2.append(param)
        print("params_to_update_2に格納:", name)
    elif name in update_param_names_3:
        param.requires_grad = True
        params_to_update_3.append(param)
        print("params_to_update_3に格納:", name)
    else:
        param.requires_grad = False
        print("勾配計算なし。学習しない:", name)

# 最適化手法の設定
optimizer = optim.SGD([
    {'params': params_to_update_1, 'lr': 1e-4},
    {'params': params_to_update_2, 'lr': 5e-4},
    {'params': params_to_update_3, 'lr': 1e-3}
], momentum=0.9)


"""モデルを学習"""

# 初期設定
# GPUが使えるか確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス: ", device)

# ネットワークをGPUへ転送
net.to(device)

# ネットワークがある程度固定であれば、高速化される
torch.backends.cudnn.benchmark = True

# エポック
num_epochs = 2

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

            # GPUが使えるならGPUにデータを転送
            inputs = inputs.to(device)
            labels = labels.to(device)

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

    

"""モデルの保存"""

save_path = './weights_fine_tuning.pth'
torch.save(net.state_dict(), save_path)

"""モデルの読み込み""""
load_path = "./weights_fine_tuning.pth"
load_weights = torch.load(load_path)
net.load_state_dict(load_weights)

# GPU上で保存された重みをCPU上でロードする場合
load_weights = torch.load(load_path, map_location={'cuda:0': 'cpu'})
net.load_state_dict(load_weights)

