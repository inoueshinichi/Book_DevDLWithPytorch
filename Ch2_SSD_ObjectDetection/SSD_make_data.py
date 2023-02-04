"""
SSD用VOD2012データセット
"""

import os
import urllib.request
import zipfile
import tarfile

if __name__ == "__main__":

    # 1) ./data/フォルダの作成
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    # 2) ./weights/フォルダの作成
    weights_dir = './weights'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    # 3) VOC2012のデータセットをここからダウンロードします
    url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    target_path = os.path.join(data_dir, "VOCtrainval_11-May-2012.tar")
    if not os.path.exists(target_path):  # 時間がかかります（約15分）
        urllib.request.urlretrieve(url, target_path)
        tar = tarfile.TarFile(target_path) # tarfile用オブジェクト
        tar.extractall(data_dir)           # tarファイルを解凍
        tar.close()

    # 4) 学習済みのSSD用のVGGのパラメータをフォルダ「weights」にダウンロード
    # MIT License
    # Copyright (c) 2017 Max deGroot, Ellis Brown
    # https://github.com/amdegroot/ssd.pytorch
    url = "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
    target_path = os.path.join(weights_dir, "vgg16_reducedfc.pth") 
    if not os.path.exists(target_path):
        urllib.request.urlretrieve(url, target_path)

    # 5) 学習済みのSSD300モデルをフォルダ「weights」にダウンロード
    # MIT License
    # Copyright (c) 2017 Max deGroot, Ellis Brown
    # https://github.com/amdegroot/ssd.pytorch
    url = "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth"
    target_path = os.path.join(weights_dir, "ssd300_mAP_77.43_v2.pth") 
    if not os.path.exists(target_path):
        urllib.request.urlretrieve(url, target_path)