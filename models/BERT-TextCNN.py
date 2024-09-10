#! -*- coding: utf-8 -*-

import json
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

maxlen = 256        # 最大长度
batch_size = 16     # 批处理
config_path = './model/config.json'     # 配置文件的路径
checkpoint_path = './model/pytorch_model.bin'    # 保存模型训练进度的路径，防止长时间训练后的模型丢失
dict_path = './model/vocab.txt'     # 字典（dictionary）数据结构的路径
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'      # 设置计算设备

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 自定义的数据集类，它继承了PyTorch的内置ListDataset类，处理和管理数据
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据
        单条格式：(文本1, 文本2, 标签id)
        """
        D = []
        f = json.load(open(filename))
        for l in f:
            text1, text2, label = l['text1'], l['text2'], l['label']
            D.append((text1, text2, float(label)))
        return D

# PyTorch数据集的填充（padding）和批处理的辅助函数，将一组非固定长度的输入样本打包成模型所需的张量格式


def collate_fn(batch):
    batch_token1_ids, batch_segment1_ids, batch_labels = [], [], []
    for text1, text2, label in batch:
        token1_ids, segment1_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
        batch_token1_ids.append(token1_ids)
        batch_segment1_ids.append(segment1_ids)
        batch_labels.append([label])
    # 使用sequence_padding函数对所有序列进行填充，保证每个样本在维度0上的长度相同，结果转换为torch tensor，并设置数据类型和设备
    batch_token1_ids = torch.tensor(sequence_padding(batch_token1_ids), dtype=torch.long, device=device)
    batch_segment1_ids = torch.tensor(sequence_padding(batch_segment1_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return (batch_token1_ids, batch_segment1_ids), batch_labels.flatten()    # 将所有标签扁平化（flatten()），将其变成一维向量，便于后续的模型训练


# 加载数据集
train_dataloader = DataLoader(MyDataset('./data/train.json'),
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(MyDataset('./data/test.json'),
                             batch_size=batch_size, collate_fn=collate_fn)


# 定义TextCNN模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True,
                                            model='ernie')
        self.relu = nn.ReLU()
        # textcnn 参数

        self.filter_number = 192    # 每一卷积层的卷积核数量
        self.kernel_number = 4      # 卷积核数目，4个卷积核
        self.hidden_size = 768      # 隐藏层每条输入序列的维度
        # textcnn
        self.conv1 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.filter_number, kernel_size=(2,),
                               padding="same", padding_mode="zeros")
        self.conv2 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros")
        self.conv3 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.filter_number, kernel_size=(5,),
                               padding="same", padding_mode="zeros")
        self.conv4 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.filter_number, kernel_size=(1,),
                               padding="same", padding_mode="zeros")
        # 定义bert上的dropout（对bert的输出进行删选）、全连接层（降维和决策）
        self.dropout = nn.Dropout(0.1)     # 正则化技术，在训练过程中有10%的概率随机将输入的某个神经元的值设为0，以减少过拟合的风险
        self.dense = nn.Linear(self.bert.configs['hidden_size'], 2)  # 全连接层将高维向量（全句信息）映射到只有两个输出节点，预测两个类别

    # 定义向前传播过程，用于文本表示学习
    def forward(self, token1_ids, segment1_ids):
        seq_output, pooled1_output = self.bert([token1_ids, segment1_ids])  # 使用预训练的BERT模型获取序列级别的输出和池化后的嵌入，整句的全局表示

        trans_embedded = torch.transpose(seq_output, dim0=1, dim1=2)

        # 卷积操作：利用text-cnn进行特征提取
        convolve1 = self.relu(self.conv1(trans_embedded))
        convolve2 = self.relu(self.conv2(trans_embedded))
        convolve3 = self.relu(self.conv3(trans_embedded))
        convolve4 = self.relu(self.conv4(trans_embedded))
        # 通过对BERT的序列输出（transposed）应用四个大小不同的卷积核，得到一组特征映射
        convolve1 = torch.transpose(convolve1, dim0=1, dim1=2)
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)
        convolve3 = torch.transpose(convolve3, dim0=1, dim1=2)
        convolve4 = torch.transpose(convolve4, dim0=1, dim1=2)
        cnn_output = torch.cat((convolve4, convolve1, convolve2, convolve3), dim=2)
        # 池化和融合：对卷积层的结果进行池化（pool_strategy=‘mean’），得到每个位置的特征向量，然后通过平均池化合并所有位置的信息
        feature_output = get_pool_emb(hidden_state=cnn_output, attention_mask=token1_ids.gt(0).long(),
                                      pool_strategy='mean')
        # dropout和全连接层
        output = self.dropout(feature_output)    # 应用dropout层来防止过拟合
        output = self.dense(output)    # 然后将处理过的特征映射进一步映射到所需的输出维度（这里是2，对应于二分类任务）
        return output   # 最后返回经过前向计算得到的预测向量


