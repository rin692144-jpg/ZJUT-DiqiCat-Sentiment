
#这次我们会加载刚才保存的最佳模型 models/sentiment_model.ckpt，不会重新训练。

import json
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops


# ============================================================
# 1. 实验基本信息
# ============================================================

print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("中文酒店评论情感分类——最佳模型评价")
print("=" * 60)

BATCH_SIZE = 64
EMBEDDING_DIM = 64

TEST_FILE = Path("data") / "test.json"
VOCAB_FILE = Path("data") / "vocab.json"
MODEL_FILE = Path("models") / "sentiment_model.ckpt"


# ============================================================
# 2. 读取测试数据和词表
# ============================================================

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)

print("\n[1] 基本信息")
print("运行设备：", ms.get_context("device_target"))
print("测试样本：", len(test_data))
print("词表大小：", VOCAB_SIZE)
print("模型文件：", MODEL_FILE)


# ============================================================
# 3. 创建测试数据集
# ============================================================

class SentimentDataset:
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        sample = self.samples[index]

        input_ids = np.array(
            sample["input_ids"],
            dtype=np.int32
        )

        label = np.array(
            sample["label"],
            dtype=np.int32
        )

        return input_ids, label

    def __len__(self):
        return len(self.samples)


test_dataset = ds.GeneratorDataset(
    SentimentDataset(test_data),
    column_names=["input_ids", "label"],
    shuffle=False
)

test_dataset = test_dataset.batch(
    BATCH_SIZE,
    drop_remainder=False
)


# ============================================================
# 4. 定义与训练时完全相同的网络
# ============================================================

class SentimentNet(nn.Cell):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.classifier = nn.Dense(
            embedding_dim,
            2
        )

    def construct(self, input_ids):

        embedded = self.embedding(input_ids)

        mask = (input_ids != 0).astype(ms.float32)
        mask = ops.expand_dims(mask, -1)

        embedded = embedded * mask

        summed = ops.sum(
            embedded,
            dim=1
        )

        lengths = ops.sum(
            mask,
            dim=1
        )

        lengths = ops.maximum(
            lengths,
            ms.Tensor(1.0, ms.float32)
        )

        pooled = summed / lengths

        logits = self.classifier(pooled)

        return logits


network = SentimentNet(
    VOCAB_SIZE,
    EMBEDDING_DIM
)


# ============================================================
# 5. 加载最佳模型
# ============================================================

param_dict = ms.load_checkpoint(
    str(MODEL_FILE)
)

ms.load_param_into_net(
    network,
    param_dict
)

network.set_train(False)

print("\n[2] 最佳模型加载成功！")


# ============================================================
# 6. 在测试集上预测
# ============================================================

all_predictions = []
all_labels = []

for input_ids, labels in test_dataset.create_tuple_iterator():

    logits = network(input_ids)

    predictions = ops.argmax(
        logits,
        dim=1
    )

    all_predictions.extend(
        predictions.asnumpy().tolist()
    )

    all_labels.extend(
        labels.asnumpy().tolist()
    )


# ============================================================
# 7. 计算混淆矩阵
# ============================================================

TP = 0
TN = 0
FP = 0
FN = 0

for true_label, predicted_label in zip(
    all_labels,
    all_predictions
):

    if true_label == 1 and predicted_label == 1:
        TP += 1

    elif true_label == 0 and predicted_label == 0:
        TN += 1

    elif true_label == 0 and predicted_label == 1:
        FP += 1

    elif true_label == 1 and predicted_label == 0:
        FN += 1


# ============================================================
# 8. 计算评价指标
# ============================================================

total = TP + TN + FP + FN

accuracy = (TP + TN) / total

precision = TP / (TP + FP) if (TP + FP) > 0 else 0

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

# 负面类别召回率：
# 所有真正负面的评论中，有多少被正确识别为负面
negative_recall = (
    TN / (TN + FP)
    if (TN + FP) > 0
    else 0
)


# ============================================================
# 9. 输出评价结果
# ============================================================

print("\n[3] 混淆矩阵")

print()
print("                 预测负面     预测正面")
print("真实负面       {:>8}     {:>8}".format(TN, FP))
print("真实正面       {:>8}     {:>8}".format(FN, TP))

print("\n[4] 评价指标")

print(
    "Accuracy：          {:.2f}%".format(
        accuracy * 100
    )
)

print(
    "Positive Precision：{:.2f}%".format(
        precision * 100
    )
)

print(
    "Positive Recall：   {:.2f}%".format(
        recall * 100
    )
)

print(
    "Positive F1-score： {:.2f}%".format(
        f1 * 100
    )
)

print(
    "Negative Recall：   {:.2f}%".format(
        negative_recall * 100
    )
)


# ============================================================
# 10. 显示部分错误案例
# ============================================================

print("\n[5] 部分错误分类案例")

error_count = 0

for i, (true_label, predicted_label) in enumerate(
    zip(all_labels, all_predictions)
):

    if true_label != predicted_label:

        print("\n----------------------------------------")

        print(
            "真实情感：",
            "正面" if true_label == 1 else "负面"
        )

        print(
            "模型预测：",
            "正面" if predicted_label == 1 else "负面"
        )

        text = test_data[i]["text"]

        # 避免特别长的评论刷满终端
        if len(text) > 150:
            text = text[:150] + "……"

        print("评论：", text)

        error_count += 1

        # 只展示前5个错误案例
        if error_count >= 5:
            break


print("\n" + "=" * 60)
print("最佳模型评价完成！")
print("浙江工业大学 · 底气猫猫虫")
print("=" * 60)