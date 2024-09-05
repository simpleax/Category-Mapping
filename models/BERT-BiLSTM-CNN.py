#! -*- coding: utf-8 -*-
# 准确率 0.98932
import json
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, text_segmentate, ListDataset, seed_everything, get_pool_emb
import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn import metrics
import tensorflow as tf
import pdb
import matplotlib.pyplot as plt

maxlen = 256
batch_size = 16
config_path = 'E:/博士小论文/基于孪生BERT网络的应急物资分类标准类目映射/代码实现/SiBert/model/config.json'
checkpoint_path = 'E:/博士小论文/基于孪生BERT网络的应急物资分类标准类目映射/代码实现/SiBert/model/pytorch_model.bin'
dict_path = 'E:/博士小论文/基于孪生BERT网络的应急物资分类标准类目映射/代码实现/SiBert/model/vocab.txt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

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
        plt.savefig("./bert_bilstm_cnn_confusion_matrix.png")
    plt.show()

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
train_dataloader = DataLoader(MyDataset('E:/博士小论文/基于孪生BERT网络的应急物资分类标准类目映射/实验数据/train.json'),
                              batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(MyDataset('E:/博士小论文/基于孪生BERT网络的应急物资分类标准类目映射/实验数据/test.json'),
                             batch_size=batch_size, collate_fn=collate_fn)

# 定义 BiLSTM-CNN 模型
class BiLSTMCNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(BiLSTMCNN, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim))
        self.conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu')
        self.global_max_pooling = tf.keras.layers.GlobalMaxPooling1D()
        self.dense = tf.keras.layers.Dense(num_classes, activation='softmax')

    def forward(self, token1_ids, segment1_ids):
        seq_output, pooled1_output = self.bert([token1_ids, segment1_ids])

        trans_embedded = torch.transpose(seq_output, dim0=1, dim1=2)

        # textcnn
        convolve2 = self.relu(self.conv2(trans_embedded))
        convolve2 = torch.transpose(convolve2, dim0=1, dim1=2)
        # lstm
        # lstm_output = self.lstm(seq_output)[0]
        feature_output = get_pool_emb(hidden_state=convolve2, attention_mask=token1_ids.gt(0).long(),
                                      pool_strategy='mean')
        output = self.dropout(feature_output)
        output = self.dense(output)
        return output


model = BiLSTMCNN().to(device)

# 定义使用的loss(Cross-Entropy loss)和optimizer
model.compile(
    loss=nn.CrossEntropyLoss(),
    optimizer=optim.AdamW(model.parameters(), lr=2e-5),
    metrics=['acc']
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
            model.save_weights('E:/博士小论文/基于孪生BERT网络的应急物资分类标准类目映射/代码实现/BERTNN2/BERT_BiLSTM_CNN/best_model.pt')
        print(f'val_acc: {test_acc:.5f}, best_val_acc: {self.best_val_acc:.5f}\n')


if True:
    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, callbacks=[evaluator])

model.load_weights('E:/博士小论文/基于孪生BERT网络的应急物资分类标准类目映射/代码实现/BERTNN2/BERT_BiLSTM_CNN/best_model.pt')
#evaluate(test_dataloader, "test")

pre = []
gro = []
for x_true, y_true in tqdm(test_dataloader):
    y_pred = model.predict(x_true).argmax(axis=1)
    pre += y_pred.tolist()
    gro += y_true.tolist()

confusion = metrics.confusion_matrix(gro, pre, labels=None)
print(confusion)
show_confusion_matrix(confusion)