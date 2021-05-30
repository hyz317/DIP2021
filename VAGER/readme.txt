文件说明：
【从清华云盘下载】base_feature.npy --1000类基本类的AlexNet.fc7层4096维特征采样，每个类别100个样本，下载链接为https://cloud.tsinghua.edu.cn/f/93d4ee8d801c4f48afe9/?dl=1
base_label.txt  --对应base_feature.npy，为每个样本的标签
base_label_meaning.txt --1000类基本类的名称
training文件夹 --包含50个新类的训练图片，每个类别10张
【暂未发布】testing文件夹 --包含50个新类的训练图片，每个类别50张，测试集将在14周左右发布
大家的任务是根据1000个基本类的特征和50个新类的少量训练图片，训练一个50分类器用于识别新类的图片
base feature的特征提取自Pytorch实现的AlxeNet预训练模型，具体使用方式可以在https://pytorch.org/vision/stable/models.html#classification找到

相关参考文献：
Matching Networks for One-shot Learning. NIPS 2016
Learning to learn: Model regression networks for easy small sample learning. ECCV 2016
Prototypical Networks for Few-shot Learning. NIPS 2017
Learning to Learn Image Classifiers with Visual Analogy. CVPR 2019
