#! -*- coding: utf-8 -*-

import json

from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt

maxlen = 100
batch_size = 16
config_path = './model/config.json'
checkpoint_path = './model/pytorch_model.bin'
dict_path = './model/vocab.txt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


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


def collate_fn(batch):
    batch_token1_ids, batch_segment1_ids, batch_token2_ids, batch_segment2_ids, batch_labels = [], [], [], [], []
    for text1, text2, label in batch:
        token1_ids, segment1_ids = tokenizer.encode(text1, maxlen=maxlen)
        token2_ids, segment2_ids = tokenizer.encode(text2, maxlen=maxlen)
        batch_token1_ids.append(token1_ids)
        batch_segment1_ids.append(segment1_ids)
        batch_token2_ids.append(token2_ids)
        batch_segment2_ids.append(segment2_ids)
        batch_labels.append([label])

    batch_token1_ids = torch.tensor(sequence_padding(batch_token1_ids), dtype=torch.long, device=device)
    batch_segment1_ids = torch.tensor(sequence_padding(batch_segment1_ids), dtype=torch.long, device=device)
    batch_token2_ids = torch.tensor(sequence_padding(batch_token2_ids), dtype=torch.long, device=device)
    batch_segment2_ids = torch.tensor(sequence_padding(batch_segment2_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.float, device=device)
    return (batch_token1_ids, batch_segment1_ids, batch_token2_ids, batch_segment2_ids), batch_labels.flatten()


# 加载数据集
train_dataloader = DataLoader(MyDataset('./data/train.json'),
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(MyDataset('./data/test.json'),
                             batch_size=batch_size, collate_fn=collate_fn)


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True,
                                            model='ernie')
        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(self.bert.configs['hidden_size'] * 3, 2)
        self.sig = nn.Sigmoid()
        self.cos = cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, token1_ids, segment1_ids, token2_ids, segment2_ids):
        _, pooled1_output = self.bert([token1_ids, segment1_ids])
        _, pooled2_output = self.bert([token2_ids, segment2_ids])

        output = self.cos(pooled1_output, pooled2_output)
        return output

