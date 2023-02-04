# パッケージのimport
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import models, transforms

print("PyTorch Version: ", torch.__version__)
print("TorchVision Version: ", torchvision.__version__)

################################################################################
# VGG-16モデル
################################################################################

""" 学習済みのVGG-16モデルをロード """

# VGG-16モデルのインスタンスを生成
use_pretained = True
net = models.vgg16(pretrained=use_pretained)
net.eval()

# モデルのネットワーク構成を出力
print(net)


"""VGG-16へ入力する画像の前処理"""
# 224 x 224 リサイズ
# 平均(0.485, 0.456, 0.406), 標準偏差(0.229, 0.224, 0.225)　規格化

# 入力画像の前処理クラス
class BaseTransform():

    def __init__(self, resize, mean, std):
        self.base_transform = transforms.Compose([
            transforms.Resize(resize),      # 画像の短辺が244になるようにリサイズ
            transforms.CenterCrop(resize),  # 画像中央を切り抜く(resize x resize)
            transforms.ToTensor(),          # Torchテンソルに変換
            transforms.Normalize(mean, std) # 色情報の規格化
        ])

    def __call__(self, img):
        return self.base_transform(img)


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
    transform = BaseTransform(resize, mean, std)
    img_transformed = transform(img) # torch.Size([3, 224, 224])
    img_transformed_transposed = img_transformed.numpy().transpose((1,2,0))

    print("img_transformed_transposed :",  img_transformed_transposed)
    img_transformed_transposed_cliped = np.clip(img_transformed_transposed, 0, 1) # 0-1にクリップ
    print("img_transformed_transposed_cliped :",  img_transformed_transposed_cliped)
    plt.imshow(img_transformed_transposed_cliped)
    plt.show()

    return img_transformed # torch.Size([3, 224, 224])

################################################################################
# VGG-16による推論
################################################################################
ILSVRC_class_index = json.load(open('./data/imagenet_class_index.json', 'r'))
print("ILSVRC_class_index :", ILSVRC_class_index)

# 推論クラス
class ILSVRCPredictor():

    def __init__(self, class_index):
        self.class_index = class_index

    def predict_max(self, out):
        maxID = np.argmax(out.detach().numpy())
        predicted_label_name = self.class_index[str(maxID)][1]
        return predicted_label_name

def inferenceImage(img_transformed):

    # ILSVRCPredictorのインスタンス
    predictor = ILSVRCPredictor(ILSVRC_class_index)

    # バッチサイズの次元を追加する
    inputs = img_transformed.unsqueeze_(0) # torch.Size([1, 3, 224, 224])

    # 推論
    out = net(inputs) # torch.Size([1, 1000])
    result = predictor.predict_max(out)

    return result


if __name__ == "__main__":

    # 1) 画像前処理
    img_transformed = preprocessImage()

    # 2) 推論
    result = inferenceImage(img_transformed)

    # 予測結果を出力する
    print("入力画像の予測結果 :", result)