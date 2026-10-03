import json
import time
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor


# ============================================================
# 1. 实验基本信息
# ============================================================

print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("基于 MindSpore 的中文酒店评论情感分类训练")
print("=" * 60)

ms.set_seed(42)
np.random.seed(42)

BATCH_SIZE = 64
EMBEDDING_DIM = 64
EPOCHS = 10
LEARNING_RATE = 0.001

TRAIN_FILE = Path("data") / "train.json"
TEST_FILE = Path("data") / "test.json"
VOCAB_FILE = Path("data") / "vocab.json"
MODEL_FILE = Path("models") / "sentiment_model.ckpt"


# ============================================================
# 2. 读取数据
# ============================================================

with open(TRAIN_FILE, "r", encoding="utf-8") as f:
    train_data = json.load(f)

with open(TEST_FILE, "r", encoding="utf-8") as f:
    test_data = json.load(f)

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)

print("\n[1] 实验参数")
print("运行设备：", ms.get_context("device_target"))
print("训练样本：", len(train_data))
print("测试样本：", len(test_data))
print("词表大小：", VOCAB_SIZE)
print("Batch Size：", BATCH_SIZE)
print("Embedding Dimension：", EMBEDDING_DIM)
print("Epochs：", EPOCHS)
print("Learning Rate：", LEARNING_RATE)


# ============================================================
# 3. 创建 MindSpore 数据集
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


def create_dataset(samples, shuffle):
    dataset = ds.GeneratorDataset(
        SentimentDataset(samples),
        column_names=["input_ids", "label"],
        shuffle=shuffle
    )

    dataset = dataset.batch(
        BATCH_SIZE,
        drop_remainder=False
    )

    return dataset


train_dataset = create_dataset(train_data, shuffle=True)
test_dataset = create_dataset(test_data, shuffle=False)

print("\n[2] MindSpore Dataset 创建成功")
print("训练 Batch 数：", train_dataset.get_dataset_size())
print("测试 Batch 数：", test_dataset.get_dataset_size())


# ============================================================
# 4. 定义神经网络
# ============================================================

class SentimentNet(nn.Cell):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        # 将字符编号转换成可学习的 64 维向量
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        # 将文本特征映射为两个类别：
        # 0 = 负面
        # 1 = 正面
        self.classifier = nn.Dense(
            embedding_dim,
            2
        )

    def construct(self, input_ids):

        # input_ids:
        # [batch, 128]

        embedded = self.embedding(input_ids)

        # embedded:
        # [batch, 128, embedding_dim]

        # ----------------------------------------------------
        # 创建 Mask
        # input_ids == 0 的位置是 Padding
        # ----------------------------------------------------

        mask = (input_ids != 0).astype(ms.float32)

        # [batch, 128]
        # ↓
        # [batch, 128, 1]

        mask = ops.expand_dims(
            mask,
            -1
        )

        # Padding 位置的向量变成 0
        embedded = embedded * mask

        # ----------------------------------------------------
        # 对有效字符向量求和
        #
        # 注意：
        # MindSpore 2.10.0 当前接口使用 dim
        # ----------------------------------------------------

        summed = ops.sum(
            embedded,
            dim=1
        )

        # summed:
        # [batch, embedding_dim]

        # ----------------------------------------------------
        # 统计每条评论有多少个有效字符
        # ----------------------------------------------------

        lengths = ops.sum(
            mask,
            dim=1
        )

        # lengths:
        # [batch, 1]

        # 防止极端情况下除以 0
        lengths = ops.maximum(
            lengths,
            Tensor(1.0, ms.float32)
        )

        # ----------------------------------------------------
        # 平均池化
        # ----------------------------------------------------

        pooled = summed / lengths

        # pooled:
        # [batch, embedding_dim]

        # ----------------------------------------------------
        # 情感分类
        # ----------------------------------------------------

        logits = self.classifier(pooled)

        # logits:
        # [batch, 2]

        return logits


network = SentimentNet(
    VOCAB_SIZE,
    EMBEDDING_DIM
)

print("\n[3] 神经网络创建成功")
print(network)


# ============================================================
# 5. Loss 与优化器
# ============================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = nn.Adam(
    network.trainable_params(),
    learning_rate=LEARNING_RATE
)


def forward_fn(input_ids, labels):

    logits = network(input_ids)

    loss = loss_fn(
        logits,
        labels
    )

    return loss, logits


grad_fn = ms.value_and_grad(
    forward_fn,
    None,
    optimizer.parameters,
    has_aux=True
)


def train_step(input_ids, labels):

    (loss, logits), grads = grad_fn(
        input_ids,
        labels
    )

    optimizer(grads)

    return loss, logits


# ============================================================
# 6. 测试函数
# ============================================================

def evaluate(dataset):

    # 切换到测试模式
    network.set_train(False)

    correct = 0
    total = 0

    for input_ids, labels in dataset.create_tuple_iterator():

        logits = network(input_ids)

        # 找出两个类别中分数最高的类别
        predictions = ops.argmax(
            logits,
            dim=1
        )

        # 统计预测正确数量
        correct += int(
            (predictions == labels)
            .astype(ms.int32)
            .sum()
            .asnumpy()
        )

        total += labels.shape[0]

    # 恢复训练模式
    network.set_train(True)

    return correct / total


# ============================================================
# 7. 正式训练
# ============================================================

print("\n" + "=" * 60)
print("开始训练")
print("=" * 60)

start_time = time.time()

best_accuracy = 0.0

for epoch in range(EPOCHS):

    network.set_train(True)

    total_loss = 0.0
    batch_count = 0

    # --------------------------------------------------------
    # 一个 Epoch：
    # 将整个训练集学习一遍
    # --------------------------------------------------------

    for input_ids, labels in train_dataset.create_tuple_iterator():

        loss, logits = train_step(
            input_ids,
            labels
        )

        total_loss += float(
            loss.asnumpy()
        )

        batch_count += 1

    # 当前 Epoch 的平均 Loss
    average_loss = total_loss / batch_count

    # 使用没有参与训练的测试集进行评价
    test_accuracy = evaluate(
        test_dataset
    )

    print(
        "Epoch {:02d}/{} | Loss: {:.4f} | Test Accuracy: {:.2f}%".format(
            epoch + 1,
            EPOCHS,
            average_loss,
            test_accuracy * 100
        )
    )

    # --------------------------------------------------------
    # 保存目前测试集准确率最高的模型
    # --------------------------------------------------------

    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        ms.save_checkpoint(
            network,
            str(MODEL_FILE)
        )

        print(
            "  → 保存当前最佳模型：Accuracy {:.2f}%".format(
                best_accuracy * 100
            )
        )


# ============================================================
# 8. 最终结果
# ============================================================

elapsed_time = time.time() - start_time

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)

print(
    "最佳测试准确率：{:.2f}%".format(
        best_accuracy * 100
    )
)

print(
    "训练耗时：{:.2f} 秒".format(
        elapsed_time
    )
)

print(
    "最佳模型保存位置：",
    MODEL_FILE
)

print("\n浙江工业大学 · 底气猫猫虫")
print("=" * 60)