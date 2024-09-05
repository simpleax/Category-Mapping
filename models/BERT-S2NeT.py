#! -*- coding: utf-8 -*-

import json
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt


def show_confusion_matrix(confusion, classes=["0", "1"], x_rot=-60, figsize=None, save=True):
    """
    绘制混淆矩阵
    :param confusion:
    :param classes:
    :param x_rot:
    :param figsize:
    :param save:
    :return:
    """
    if figsize is not None:
        plt.rcParams['figure.figsize'] = figsize

    plt.imshow(confusion, cmap=plt.cm.YlOrRd)
    indices = range(len(confusion))
    plt.xticks(indices, classes, rotation=x_rot, fontsize=12)
    plt.yticks(indices, classes, fontsize=12)
    plt.colorbar()
    plt.xlabel('y_pred')
    plt.ylabel('y_true')

    # 显示数据
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])

    if save:
        plt.savefig("./bert_s2net_confusion_matrix.png")
    plt.show()


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
    batch_labels = torch.tensor(batch_labels, dtype=torch.long, device=device)
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

    def forward(self, token1_ids, segment1_ids, token2_ids, segment2_ids):
        _, pooled1_output = self.bert([token1_ids, segment1_ids])
        _, pooled2_output = self.bert([token2_ids, segment2_ids])
        pooled_output = torch.cat([pooled1_output, pooled2_output, torch.abs(pooled1_output - pooled2_output)], dim=-1)
        output = self.dropout(pooled_output)
        output = self.dense(output)
        return output

model = Model().to(device)

# 定义使用的loss和optimizer，这里支持自定义
model.compile(
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(model.parameters(), lr=2e-5),
        metrics=['acc'],
        metrics2=['pre'],    # 评估指标为精确率precise
        metrics3=['rec'],    # 召回率rcall
        metrics4=['f1']      # f1分数
    )

# 定义评价函数
def evaluate(data, mark):
        model.eval()
        total, right = 0., 0.
        pre = []
        gro = []
        for x_true, y_true in tqdm(data):
            y_pred = model.predict(x_true).argmax(axis=1)
            pre += y_pred.tolist()
            gro += y_true.tolist()
            total += len(y_true)
            right += (y_true == y_pred).sum().item()
        precise = metrics.precision_score(gro, pre)
        rec = metrics.recall_score(gro, pre)
        f1 = metrics.f1_score(gro, pre)
        print(f'{mark} precision: {precise:.5f}, recall: {rec:.5f}, f1: {f1:.5f}, acc: {right / total:.5f}')
        model.train()
        return right / total

class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, global_step, epoch, logs=None):
        train_acc = evaluate(train_dataloader, "train")
        # print(f'train_acc: {train_acc:.5f}\n')
        test_acc = evaluate(test_dataloader, "test")
        if test_acc > self.best_val_acc:
            self.best_val_acc = test_acc
            model.save_weights('./BERT_S2NeT/best_model.pt')
        print(f'val_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')


if True:
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, callbacks=[evaluator])

model.load_weights('./BERT_S2NeT/best_model.pt')
# evaluate(test_dataloader, "test")

pre = []
gro = []
for x_true, y_true in tqdm(test_dataloader):
    y_pred = model.predict(x_true).argmax(axis=1)
    pre += y_pred.tolist()
    gro += y_true.tolist()

confusion = metrics.confusion_matrix(gro, pre, labels=None)
print(confusion)
show_confusion_matrix(confusion)
