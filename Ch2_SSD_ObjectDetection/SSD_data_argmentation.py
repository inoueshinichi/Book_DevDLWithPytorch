"""オーグメンテーションを含む画像の前処理
"""

from numpy import random
import types
import cv2
import numpy as np

# 1) torchのtransformsオブジェクトを用いて画像を水増しするインターフェースクラス
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels



# 2) torch.transformsにlambda式インターフェースのクラス
class Lambda(object):
    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd
    
    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)



# 3) 画像の画素値を整数から実数に変換するクラス
class ConvertInt2Float(object):
    def __call__(self, img, boxes, labels):
        return img.astype(np.float32), boxes, labels



# 4) 各画素から平均値を減算するクラス
class SubtractMean(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)
        
    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), boxes, labels



# 5) バウンディングボックスを正規化値から絶対数値に変換するクラス
class ToAbsoluteCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        width, height, channels = img.shape # 縦　横　チャンネル
        boxes[:, 0] *= width
        boxes[:, 1] *= width
        boxes[:, 2] *= height
        boxes[:, 3] *= height
        return img, boxes, labels



# 6) バウンディングボックスを絶対数値から正規化値に変換するクラス
class ToPercentCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        width, height, channels = img.shape # 縦　横　チャンネル
        boxes[:, 0] /= width
        boxes[:, 1] /= width
        boxes[:, 2] /= height
        boxes[:, 3] /= height
        return img, boxes, labels



# 7) 画像をリサイズするするクラス
class Resize(object):
    def __init__(self, size=300):
        self.size = size
        
    def __call__(self, img, boxes=None, labels=None):
        img = cv2.resize(img, (self.size, self.size))
        return img, boxes, labels



# 8) ランダムに彩度(saturation)を変化させるクラス
class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 1] *= random.uniform(self.lower, self.upper) # チャネル１が彩度
        return img, boxes, labels



# 9) ランダムに色彩(Hue)を変化させるクラス
class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta
        
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0][img[:, :, 0] > 360] = -360.0
            img[:, :, 0][img[:, :, 0] < 0] = +360.0
        return img, boxes, labels



# 10) ランダムに輝度を変化させるクラス
class RandomLightingNoise(object):
    def __init__(self, delta=32):
        assert(delta >= 0)
        assert(delta <= 255)
        self.delta = delta
    
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, boxes, labels



# 11) ランダムにコントラストを変化させるクラス
class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."
        
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha # 単純な線形濃度変換(コントラスト変換といえるのか？)
        return img, boxes, labels



# 12) ランダムで画像を反転させるクラス
class RandomMirror(object):
    def __call__(self, img, boxes, labels):
        if random.randint(2):
            img = img[:, :-1]
            _, width, _ = img.shape # (c, w, h)
            boxes[0:1:-1] = width - boxes[0:1:-1] # boxes[xmin ymin, xmax, ymax, index]
        return img, boxes, labels



# 13) ランダムにチャンネルを入れ替えるクラス
class RandomBrightness(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            img = img[:, :, swaps]            
        return img, boxes, labels



# 14) データ表現の変更(HSV<->BGR)のクラス
class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, img, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return img, boxes, labels



# 15) torchテンソルからopencvマットに変換するクラス
class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels



# 16) opencvマットからtorchテンソルに変換するクラス
class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels



# 17) ランダムに画像をクロップする機能とクラス
# intersect値(重なった面積)を求める
def intersect(a_box, b_boxes):
    # box : [xmin, ymin, xmax, ymax]
    # 対象AのboundingBoxとその他全てのboundingBoxを比較して、(xmin, ymin), (xmax, ymax)を求める
    xy_max = np.minimum(a_box[2:], b_boxes[:, 2:])
    xy_min = np.maximum(a_box[:2], b_boxes[:, :2])
    inter = np.clip((xy_max - xy_min), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

# ジャッカード係数を求める
def jaccard_numpy(a_box, b_boxes):
    # A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    # a_box   : single box Shape: [xmin, ymin, xmax, ymax]
    # b_boxs  : Multiple bounding boxes, Shape: [num_boxes,4] 
    inners = intersect(a_box, b_boxes)
    a_area = (a_box[2] - a_box[0]) * (a_box[3] - a_box[0])
    b_areas = (b_boxes[:, 2] - b_boxes[:, 0]) * (b_boxes[:, 3] * b_boxes[:, 0])
    unions = a_area + b_areas - inners
    return inners / unions



class RandomCrop(object):
    def __init__(self):
        self.sample_options = (
            None,         # 画像全体を利用
            (0.1, None),  # Jaccard係数 0.1 - max:inf
            (0.3, None),  # Jaccard係数 0.3 - max:inf
            (0.4, None),  # Jaccard係数 0.4 - max:inf
            (0.7, None),  # Jaccard係数 0.7 - max:inf
            (0.9, None),  # Jaccard係数 0.9 - max:inf
            (None, None)  # 完全ランダムクロップ
        )
        
    def __call__(self, img, boxes=None, labels=None):
        height, width, _ = img.shape
        
        while True:
            mode = random.choice(self.sample_options)
        
            # 画像全体を利用
            if mode is None:
                return img, boxes, labels
        
            # Jaccard係数の範囲
            iou_min, iou_max = mode 
            if iou_min is None:
                iou_min = float('-inf')
            if iou_max is None:
                iou_max = float('inf')
            
            # ランダムに最大50枚のクロップ画像を取得
            for _ in range(50):
                current_img = img
                w, h = random.uniform(0.3 * width, width), random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2: # 横長or縦長が強すぎる場合は、現在のクロップ処理をスルー
                    continue
            
                # クロップ領域のleft,topをランダムに決定
                left, top = random.uniform(width - w), random.uniform(height - h)
            
                # クロップ領域とアノテーションのbboxの間でJaccard係数を算出
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                overlap = jaccard_numpy(rect, boxes)
            
                # Jaccard係数の最大最小を確認して、しきい値を満たさない場合、現在のクロップ処理をスルー
                if overlap.min() < iou_min and overlap.max() > iou_max: # この条件が理解できてない
                    continue
            
                # クロップ画像
                current_img = current_img[rect[1]:rect[3], rect[0]:rect[2], :]
            
                # クロップ領域にbboxesの重心が含まれているかチェック
                bboxes_centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                m1 = (bboxes_centers[:, 0] > rect[0]) * (bboxes_centers[:, 1] > rect[1])
                m2 = (bboxes_centers[:, 0] < rect[2]) * (bboxes_centers[:, 1] < rect[3])
                masks = m1 * m2
            
                # bboxesの重心が1つもクロップ領域に含まれない場合、現在のクロップ処理をスルー
                if not masks.any():
                    continue
                
                # 重心がクロップ領域の入っているbboxだけ取り出す
                current_bboxes = boxes[masks, :].copy()
                current_labels = labels[masks]
            
                # bboxのleft, top, right, bottomをクロップ領域内に制限する
                current_bboxes[:, :2] = np.maximum(current_bboxes[:, :2], rect[:2])
                current_bboxes[:, :2] -= rect[:2] # (x_min, y_min) -> (0, 0)
                current_bboxes[:, 2:] = np.minimum(current_bboxes[:, 2:], rect[2:])
                current_bboxes[:, 2:] -= rect[:2] # (x_max, y_max) -> (width-1, height-1)
            
                return current_img, current_bboxes, current_labels


# 18) コントラスト->彩度->色彩->コントラストの順番にランダムに値を変換するクラス
class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()
        
    def __call__(self, img, boxes, labels):
        im = img.copy()
        #im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)


# 19) 拡張画像を提供するクラス
class Expand(object):
    def __init__(self, mean):
        self.mean = mean
        
    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels
        
        height, width, channels = img.shape
        
        # 拡大率
        ratio = random.uniform(1, 4)
        
        # 拡大画像内で原画像を埋め込むときの(left, top)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        
        # 拡張画像
        expand_img = np.zeros(
            (int(height * ratio), int(width * ratio), channels),
            dtype=img.dtype)
        expand_img[:,:,:] = self.mean # 拡張領域は平均値で埋める
        expand_img[int(top):int(top + height), int(left):int(left + width)] = img
        img = expand_img
        
        # bboxのleft, top, right, bottomを編集
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))
        
        return img, boxes, labels


"""前処理クラス
"""
class DataTransform():
    """
        画像とアノテーションの前処理クラス。学習と推論で異なる動作をする。
        画像サイズ : 300 x 300
        学習時はデータオーグメンテーションをする。
        
        Attributes
        ----------
        input_size : int
            リサイズ先の画像の大きさ
        color_mean : (B, G, R)
            各色チャネルの平均値
    """
    
    def __init__(self, input_size, color_mean):
        self.data_transform = {
            "train" : Compose([
                ConvertInt2Float(),      # int -> float
                ToAbsoluteCoords(),      # bbox int -> float
                PhotometricDistort(),    # コントラスト->彩度->色彩->コントラストの順序でランダムに値を変換
                Expand(color_mean),      # 画像サイズ拡張
                RandomCrop(),            # クロップ
                RandomMirror(),          # 水平方向反転
                ToPercentCoords(),       # bbox float -> int
                Resize(input_size),      # リサイズ 上記のToPersentCoords()でbboxを0~1に規格化しているからOK
                SubtractMean(color_mean) # 平均値を引く
            ]),
            "val" : Compose([
                ConvertInt2Float(),      #int -> float
                Resize(input_size),      # リサイズ 上記のToPersentCoords()でbboxを0~1に規格化しているからOK
                SubtractMean(color_mean) # 平均値を引く
            ])
        }
        
    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)


if __name__ == "__main__":

    # 学習データ
    from SSD_make_dataset import make_datapath_list
    rootpath = "./data/VOCdevkit/VOC2012/"
    train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(rootpath)

    # 1. 画像読み込み
    image_file_path = train_img_list[0]
    img = cv2.imread(image_file_path)
    height, width, channels = img.shape
    print("path: {}\nheight: {}, width: {}, channels: {}".format(image_file_path, 
                                                                 height, 
                                                                 width, 
                                                                 channels))

    # 2. アノテーションをリストに変換
    voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
    from SSD_convert_xml_annotation_data import Anno_xml2list
    transform_anno = Anno_xml2list(voc_classes)
    anno_list = transform_anno(train_anno_list[0], width, height)
    print("voc-class's Annotation: \n{}".format(anno_list))

    # 3.原画像の表示
    import matplotlib.pyplot as plt
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # 4.前処理クラスの作成
    color_mean = (104, 117, 123) # VOCデータセットの(BGR)平均
    input_size = 300
    transform = DataTransform(input_size, color_mean)

    # 5.train画像の表示
    phase = 'train'
    img_transformed, boxes, labels = transform(img, phase, anno_list[:, :4], anno_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()

