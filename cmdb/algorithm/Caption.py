import torch as t
import torchvision as tv
from cmdb.algorithm.Models import CaptionModel
from PIL import Image
import pandas as pd
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class Config:
    caption_data_path = './media/caption_11_2.pth'  # 经过预处理后的人工描述信息
    img_path = './img_t//'
    # img_path='/mnt/ht/aichallenger/raw/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
    img_feature_path = 'results_10_07.pth'  # 所有图片的features,20w*2048的向量
    scale_size = 300
    img_size = 224
    batch_size = 12
    shuffle = True
    num_workers = 0
    rnn_hidden = 256
    embedding_dim = 256
    num_layers = 2
    share_embedding_weights = True

    prefix = 'checkpoints/caption'  # 模型保存前缀

    env = 'caption'
    plot_every = 10
    debug_file = '/tmp/debugc'

    model_ckpt = './media/caption_1124_2204'  # 模型断点保存路径
    lr = 1e-3
    use_gpu = True
    epoch = 1000



def generate(image_path,name):
    opt = Config()
    # 数据预处理
    data = t.load(opt.caption_data_path, map_location=lambda s, l: s)
    word2ix, ix2word, id2ix = data['word2ix'], data['ix2word'], data['id2ix']

    name = name[4:]
    id = id2ix[name]
    data = pd.read_excel('./media/INbreast_修改.xls')
    row = data.iloc[id]
    classify = row['Bi-Rads']
    caption = row['caption']
    label = []

    if row['Mass '] is not np.nan:
        label.append('Mass')
    if row['Micros'] is not np.nan:
        label.append('Micros_calcification')
    if row['Multiple'] is not np.nan:
        label.append('Multiple')
    if row['Distortion'] is not np.nan:
        label.append('Distortion')
    if row['Asymmetry'] is not np.nan:
        label.append('Asymmetry')

    # with open('./media/record.txt', 'r', encoding='gbk') as file_to_read:
    #     lines = file_to_read.read()
    #     txt = lines.replace("\n", "")
    # record = eval(txt)
    # caption = record[id]['caption']

    normalize = tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transforms = tv.transforms.Compose([
        tv.transforms.Resize(opt.scale_size),
        tv.transforms.CenterCrop(opt.img_size),
        tv.transforms.ToTensor(),
        normalize
    ])
    img = Image.open(image_path).convert('RGB')
    img = transforms(img).unsqueeze(0)


    model = CaptionModel(opt, word2ix, ix2word)
    model = model.load(opt.model_ckpt).eval()
    if opt.use_gpu:
        model.cuda()

    results, vector = model.generate(img)
    return results, caption, classify, label

# def get_original_text(name):
#     report =
#     return text