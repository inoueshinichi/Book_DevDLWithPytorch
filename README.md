# Pytorchによる発展ディープラーニング</br>
https://book.mynavi.jp/ec/products/detail/id=104855</br>

第1章　画像分類と転移学習（VGG）</br>
1.1　学習済みのVGGモデルを使用する方法</br>
1.2　PyTorchによるディープラーニング実装の流れ</br>
1.3　転移学習の実装</br>
1.4　Amazon AWSのクラウドGPUマシンを使用する方法</br>
1.5　ファインチューニングの実装</br>
</br>
第2章　物体検出（SSD）</br>
2.1　物体検出とは</br>
2.2　Datasetの実装</br>
2.3　DataLoaderの実装</br>
2.4　ネットワークモデルの実装</br>
2.5　順伝搬関数の実装</br>
2.6　損失関数の実装</br>
2.7　学習と検証の実施</br>
2.8　推論の実施</br>
</br>
第3章　セマンティックセグメンテーション（PSPNet）</br>
3.1　セマンティックセグメンテーションとは</br>
3.2　DatasetとDataLoaderの実装</br>
3.3　PSPNetのネットワーク構成と実装</br>
3.4　Featureモジュールの解説と実装</br>
3.5　Pyramid Poolingモジュールの解説と実装</br>
3.6　Decoder、AuxLossモジュールの解説と実装</br>
3.7　ファインチューニングによる学習と検証の実施</br>
3.8　セマンティックセグメンテーションの推論</br>
</br>
第4章　姿勢推定（OpenPose）</br>
4.1　姿勢推定とOpenPoseの概要</br>
4.2　DatasetとDataLoaderの実装</br>
4.3　OpenPoseのネットワーク構成と実装</br>
4.4　Feature、Stageモジュールの解説と実装</br>
4.5　TensorBoardXを使用したネットワークの可視化手法</br>
4.6　OpenPoseの学習</br>
4.7　OpenPoseの推論</br>
</br>
第5章　GANによる画像生成（DCGAN、Self-Attention GAN）</br>
5.1　GANによる画像生成のメカニズムとDCGANの実装</br>
5.2　DCGANの損失関数、学習、生成の実装</br>
5.3　Self-Attention GANの概要</br>
5.4　Self-Attention GANの学習、生成の実装</br>
</br>
第6章　GANによる異常検知（AnoGAN、Efficient GAN）</br>
6.1　GANによる異常画像検知のメカニズム</br>
6.2　AnoGANの実装と異常検知の実施</br>
6.3　Efficient GANの概要</br>
6.4　Efficient GANの実装と異常検知の実施</br>
</br>
第7章　自然言語処理による感情分析（Transformer）</br>
7.1　形態素解析の実装（Janome、MeCab＋NEologd）</br>
7.2　torchtextを用いたDataset、DataLoaderの実装</br>
7.3　単語のベクトル表現の仕組み（word2vec、fastText）</br>
7.4　word2vec、fastTextで日本語学習済みモデルを使用する方法</br>
7.5　IMDb（Internet Movie Database）のDataLoaderを実装</br>
7.6　Transformerの実装（分類タスク用）</br>
7.7　Transformerの学習・推論、判定根拠の可視化を実装</br>
</br>
第8章　自然言語処理による感情分析（BERT）</br>
8.1　BERTのメカニズム</br>
8.2　BERTの実装</br>
8.3　BERTを用いたベクトル表現の比較（bank：銀行とbank：土手）</br>
8.4　BERTの学習・推論、判定根拠の可視化を実装</br>
</br>
第9章　動画分類（3DCNN、ECO）</br>
9.1　動画データに対するディープラーニングとECOの概要</br>
9.2　2D Netモジュール（Inception-v2）の実装</br>
9.3　3D Netモジュール（3DCNN）の実装</br>
9.4　Kinetics動画データセットをDataLoaderに実装</br>
9.5　ECOモデルの実装と動画分類の推論実施</br>