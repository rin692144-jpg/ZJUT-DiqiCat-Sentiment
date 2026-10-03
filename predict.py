import json
from pathlib import Path

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


# ============================================================
# 1. 基本参数
# ============================================================

VOCAB_FILE = Path("data") / "vocab.json"
MODEL_FILE = Path("models") / "sentiment_model.ckpt"

MAX_LENGTH = 128
EMBEDDING_DIM = 64


print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("基于 MindSpore 的中文情感分类预测系统")
print("=" * 60)


# ============================================================
# 2. 加载词表
# ============================================================

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

VOCAB_SIZE = len(vocab)

PAD_ID = vocab["<PAD>"]
UNK_ID = vocab["<UNK>"]


# ============================================================
# 3. 文本编码
# ============================================================

def encode_text(text):

    ids = [
        vocab.get(char, UNK_ID)
        for char in text
    ]

    # 截断
    ids = ids[:MAX_LENGTH]

    # Padding
    if len(ids) < MAX_LENGTH:
        ids += [PAD_ID] * (
            MAX_LENGTH - len(ids)
        )

    return np.array(
        [ids],
        dtype=np.int32
    )


# ============================================================
# 4. 定义与训练时完全相同的模型
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

        embedded = self.embedding(
            input_ids
        )

        mask = (
            input_ids != 0
        ).astype(ms.float32)

        mask = ops.expand_dims(
            mask,
            -1
        )

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
            ms.Tensor(
                1.0,
                ms.float32
            )
        )

        pooled = summed / lengths

        logits = self.classifier(
            pooled
        )

        return logits


# ============================================================
# 5. 创建模型并加载训练好的参数
# ============================================================

network = SentimentNet(
    VOCAB_SIZE,
    EMBEDDING_DIM
)

param_dict = ms.load_checkpoint(
    str(MODEL_FILE)
)

ms.load_param_into_net(
    network,
    param_dict
)

network.set_train(False)

print("\n模型加载成功！")
print("模型：", MODEL_FILE)
print("词表大小：", VOCAB_SIZE)


# ============================================================
# 6. 情感预测函数
# ============================================================

def predict(text):

    encoded = encode_text(text)

    input_tensor = ms.Tensor(
        encoded,
        ms.int32
    )

    logits = network(
        input_tensor
    )

    # Softmax 将模型输出转换成概率
    probabilities = ops.softmax(
        logits,
        axis=1
    )

    probabilities = (
        probabilities
        .asnumpy()[0]
    )

    negative_probability = float(
        probabilities[0]
    )

    positive_probability = float(
        probabilities[1]
    )

    if positive_probability >= negative_probability:
        sentiment = "正面"
    else:
        sentiment = "负面"

    return (
        sentiment,
        negative_probability,
        positive_probability
    )


# ============================================================
# 7. 预设测试
# ============================================================

test_sentences = [

    "这家酒店环境很好，房间干净，服务人员也非常热情。",

    "房间非常脏，隔音特别差，服务态度也不好。",

    "酒店位置不错，但是房间设施比较老旧。",

    "浙江工业大学底气猫猫虫觉得这家酒店真不错！",

    "浙江工业大学底气猫猫虫觉得这里又脏又差，再也不会来了！"
]


print("\n" + "=" * 60)
print("预设文本测试")
print("=" * 60)

for text in test_sentences:

    sentiment, negative_prob, positive_prob = predict(
        text
    )

    print("\n文本：", text)

    print(
        "预测结果：",
        sentiment
    )

    print(
        "负面概率：{:.2f}%".format(
            negative_prob * 100
        )
    )

    print(
        "正面概率：{:.2f}%".format(
            positive_prob * 100
        )
    )


# ============================================================
# 8. 用户自己输入文本
# ============================================================

print("\n" + "=" * 60)
print("进入自定义预测模式")
print("输入 q 可以退出")
print("=" * 60)

while True:

    text = input(
        "\n请输入一段酒店评论："
    ).strip()

    if text.lower() == "q":
        break

    if not text:
        print("输入不能为空。")
        continue

    sentiment, negative_prob, positive_prob = predict(
        text
    )

    print("\n预测结果：", sentiment)

    print(
        "负面概率：{:.2f}%".format(
            negative_prob * 100
        )
    )

    print(
        "正面概率：{:.2f}%".format(
            positive_prob * 100
        )
    )


print("\n" + "=" * 60)
print("预测程序结束")
print("浙江工业大学 · 底气猫猫虫")
print("=" * 60)