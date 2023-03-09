# -*- coding: utf-8 -*-
_author_ = "March_H"
'''
仅提供了关于MDFEND文本文件的dataloader，对于另一部分图像的特征提取还未添加
'''

import torch
import torch.nn as nn
import os
import json
import random
from transformers import BertTokenizer
import numpy as np
import cv2


# 构建了训练集、测试集以及验证集
# 建议在run函数中只进行训练以及验证，而测试单独写文件进行
# 注意在训练过程中只需在最开头运行一次该函数，否侧会破坏原始数据
def build_datasets():
    # 假新闻共4488条，真新闻共4640条，一共9128条，取70%（6390）作为训练集，15%（1369）作为测试集和验证集
    data_path = "./datasets/weibo21/"
    fake_data_path = data_path + "fake_release_all.json"
    true_data_path = data_path + "real_release_all.json"

    news = []
    # 读取虚假新闻
    with open(fake_data_path, "r", encoding='utf-8') as f:
        fake_data = f.readlines()
    for i in fake_data:
        i = json.loads(i)
        x = {}
        x['content'], x['label'], x['category'] = i['content'], '1', i['category']
        x['img'] = data_path + "imgs/" + i['id'] + '.jpg'
        news.append(x)
    # 读取真实新闻
    with open(true_data_path, "r", encoding='utf-8') as f:
        true_data = f.readlines()
    for i in true_data:
        i = json.loads(i)
        x = {}
        x['content'], x['label'], x['category'] = i['content'], '0', i['category']
        x['img'] = data_path + "imgs/" + i['id'] + '.jpg'
        news.append(x)
    # 随机打乱
    random.shuffle(news)
    # 构建训练数据
    with open(data_path + "train.txt", "w", encoding="utf-8") as f:
        f.truncate()  # 清空文件
        for i in range(6390):
            f.write(news[i]['content'] + '\t' +
                    news[i]['img'] + '\t' +
                    news[i]['label'] + '\t' +
                    news[i]['category'] + '\n')
    # 构建验证数据
    with open(data_path + "valid.txt", "w", encoding="utf-8") as f:
        f.truncate()  # 清空文件
        for i in range(6390, 6390 + 1369):
            f.write(news[i]['content'] + '\t' +
                    news[i]['img'] + '\t' +
                    news[i]['label'] + '\t' +
                    news[i]['category'] + '\n')
    # 构建测试数据
    with open(data_path + "test.txt", "w", encoding="utf-8") as f:
        f.truncate()  # 清空文件
        for i in range(6390 + 1369, 6390 + 1369 + 1369):
            f.write(news[i]['content'] + '\t' +
                    news[i]['img'] + '\t' +
                    news[i]['label'] + '\t' +
                    news[i]['category'] + '\n')


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)  # 基于给出的vocab文件切词
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


class MDRMFNDDatasets(nn.Module):
    def __init__(self, TextPath, vocab_file, max_len, category_dict, mode='train'):
        super(MDRMFNDDatasets, self).__init__()

        # 请将传入类的初始化参数放在以下
        self.TextPath = TextPath
        self.content = []
        self.content_token_ids = []
        self.content_masks = []
        self.imgs = []
        self.label = []
        self.category = []
        self.vocab_file = vocab_file
        self.max_len = max_len
        self.category_dict = category_dict

        # 下面进行训练集的构建
        if mode == 'train':
            # 下面的代码进行文本数据、图像路径检索、标签以及分类的读取
            file_path = TextPath + "train.txt"
            pass
        elif mode == 'valid':
            file_path = TextPath + "valid.txt"
            pass
        elif mode == 'test':
            file_path = TextPath + "test.txt"
            pass
        with open(file_path, "r", encoding='utf-8') as f:
            data = f.readlines()

        # 将构建好的数据集存进当前类中
        for x in data:
            x = x.strip()
            x = x.split("\t")
            self.content.append(x[0])
            self.imgs.append(x[1])
            self.label.append(x[2])
            self.category.append(x[3])

        # 下面对文本数据进行分词以及映射处理
        self.content_token_ids, self.content_masks = word2input(self.content, self.vocab_file, self.max_len)

    def __getitem__(self, item):
        # 采用torch自带的Dataloader方式加载数据集，将需要取出的数据通过该函数返回即可
        # 将返回类型设置为sample
        # 暂规定Key如下：
        # TextData：存储文本信息
        # Mask：存储mask，mask为0表示当前位为pad
        # ImageData：存储相应的图像信息
        # Label：记录该数据是否为假新闻，如果为1，则为假新闻，否则为真新闻
        # category：存储该图像的类别
        # 分类类别的标号详见run.py
        sample = {}

        # 设置文本数据
        text_data = self.content_token_ids[item]
        text_data = torch.as_tensor(text_data, dtype=torch.float32)
        sample['TextData'] = text_data

        # 设置图像数据
        # TODO:请在这里读取图像数据并将图像数据转化为tentor，存入sample中

        # 设置标签数据
        label_data = self.label[item]
        label_data = torch.as_tensor(label_data, dtype=torch.float32)
        sample['Label'] = label_data

        # 设置分类数据
        category_data = self.category_dict[self.category[item]]
        category_data = torch.as_tensor(category_data, dtype=torch.float32)
        sample['Category'] = category_data

        return sample
        pass
