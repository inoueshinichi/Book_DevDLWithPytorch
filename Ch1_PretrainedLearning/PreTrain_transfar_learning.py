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

################################################################################
# 画像前処理動作を確認
################################################################################
def preprocessImage():
    
    # 画像読み込み
    image_file_path = "./data/goldenretriever-3724972_640.jpg"
    img = Image.open(image_file_path)

    # 原画像を表示
    plt.imshow(img)
    plt.show()

    # 前処理と処理後の画像を表示
    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = ImageTransform(resize, mean, std)
    img_transformed = transform(img, 'train') # torch.Size([3, 224, 224])
    img_transformed_transposed = img_transformed.numpy().transpose((1,2,0))

    print("img_transformed_transposed :",  img_transformed_transposed)
    img_transformed_transposed_cliped = np.clip(img_transformed_transposed, 0, 1) # 0-1にクリップ
    print("img_transformed_transposed_cliped :",  img_transformed_transposed_cliped)
    plt.imshow(img_transformed_transposed_cliped)
    plt.show()

    return img_transformed # torch.Size()

################################################################################
# データセット
################################################################################

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

################################################################################
# 出力層改良モデルVGG-16
################################################################################
def modefiedModelVGG16():

    # 学習済みモデルVGG-16をロード
    use_pretained = True
    net = models.vgg16(pretrained=use_pretained)

    # 出力層の出力ユニットをアリとハチの2つに付け替える
    net.classifier[6] = nn.Linear(in_features=4096, out_features=2)

    # 訓練モードに設定
    net.train()
    print('ネットワーク設定完了: 学習済み重みを読み込み、訓練モードに設定しました。')

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    """ 最適化手法の設定 """
    # 転移学習で学習させるパラメータを変数params_to_updateに格納する
    params_to_update = []

    # 学習させるパラメータ名
    update_param_names = ["classifier.6.weight", "classifier.6.bias"]

    # 学習させるパラメータ以外は勾配計算をなくし、変化しないように設定
    for name, param in net.named_parameters():
        if name in update_param_names:
            param.requires_grad = True
            params_to_update.append(param)
            print(name)
        else:
            param.requires_grad = False
    
    # params_to_updateの中身を確認
    print('-------------------')
    print(params_to_update)


    # Momentum SGD
    optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)

    return criterion, optimizer, net # 損失関数, 最適化手法

################################################################################
# 学習と検証
################################################################################
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
            for inputs, labels in tqdm(dataloader_dict[phase]):

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


if __name__ == "__main__":

    # 1) 前処理
    #preprocessImage()

    # 2) データセットファイル
    #make_datapath_list()

    # 3) データセット作成
    train_dataset, val_dataset = makeDataset()

    # 4) データローダー
    dataloader_dict = dataLoader(train_dataset, val_dataset)

    # 転移学習モデル(VGG-16), 損失関数, 最適化手法設定
    criterion, optimizer, net = modefiedModelVGG16()

    # 学習と検証
    num_epochs = 2
    train_model(net, dataloader_dict, criterion, optimizer, num_epochs = num_epochs)

