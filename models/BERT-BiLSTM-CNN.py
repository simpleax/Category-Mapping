#! -*- coding: utf-8 -*-

import json
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

maxlen = 256
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
    batch_token1_ids, batch_segment1_ids, batch_labels = [], [], []
    for text1, text2, label in batch:
        token1_ids, segment1_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
        batch_token1_ids.append(token1_ids)
        batch_segment1_ids.append(segment1_ids)
        batch_labels.append([label])

    batch_token1_ids = torch.tensor(sequence_padding(batch_token1_ids), dtype=torch.long, device=device)
    batch_segment1_ids = torch.tensor(sequence_padding(batch_segment1_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
    return (batch_token1_ids, batch_segment1_ids), batch_labels.flatten()


# 加载数据集
train_dataloader = DataLoader(MyDataset('./data/train.json'),
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(MyDataset('./data/test.json'),
                             batch_size=batch_size, collate_fn=collate_fn)


# 定义LSTM-CNN模型结构
class Model(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path, with_pool=True,
                                            model='ernie')
        self.relu = nn.ReLU()
        # lstm
        self.lstm = nn.LSTM(768, 768, num_layers=1, bidirectional=False, batch_first=True)
        self.relu = nn.ReLU()
        # lstmcnn参数
        self.filter_number = 192
        self.kernel_number = 4
        self.hidden_size = 768
        #lstmcnn
        self.conv2 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.filter_number, kernel_size=(3,),
                               padding="same", padding_mode="zeros")

        self.dropout = nn.Dropout(0.1)
        self.dense = nn.Linear(192, 2)

    def forward(self, token1_ids, segment1_ids):
        seq_output, pooled1_output = self.bert([token1_ids, segment1_ids])
        # lstm
        lstm_output = self.lstm(seq_output)[0]

        trans_embedded = torch.transpose(lstm_output, dim0=1, dim1=2)

        # cnn
        convolve2 = self.relu(self.conv2(trans_embedded))
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)

        feature_output = get_pool_emb(hidden_state=convolve2, attention_mask=token1_ids.gt(0).long(),
                                      pool_strategy='mean')
        output = self.dropout(feature_output)
        output = self.dense(output)
        return output

