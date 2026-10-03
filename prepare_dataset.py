import csv
import json
import random
from pathlib import Path

print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("中文情感分类——训练集与测试集准备")
print("=" * 60)

# --------------------------------------------------
# 1. 基本参数
# --------------------------------------------------

DATA_FILE = Path("data") / "ChnSentiCorp_htl_all.csv"
VOCAB_FILE = Path("data") / "vocab.json"

MAX_LENGTH = 128
TEST_RATIO = 0.2
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# --------------------------------------------------
# 2. 读取词表
# --------------------------------------------------

with open(VOCAB_FILE, "r", encoding="utf-8") as f:
    vocab = json.load(f)

PAD_ID = vocab["<PAD>"]
UNK_ID = vocab["<UNK>"]

print("\n[1] 词表大小：", len(vocab))
print("[2] 最大文本长度：", MAX_LENGTH)


# --------------------------------------------------
# 3. 文本编码函数
# --------------------------------------------------

def encode_text(text):
    """
    将中文文本转换成固定长度的整数序列。

    过长：截断到 MAX_LENGTH
    过短：使用 <PAD> 补到 MAX_LENGTH
    """

    ids = [
        vocab.get(char, UNK_ID)
        for char in text
    ]

    # 截断
    ids = ids[:MAX_LENGTH]

    # Padding
    if len(ids) < MAX_LENGTH:
        ids += [PAD_ID] * (MAX_LENGTH - len(ids))

    return ids


# --------------------------------------------------
# 4. 读取数据，并按照正负类别分别保存
# --------------------------------------------------

positive_samples = []
negative_samples = []

with open(DATA_FILE, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if not row["review"] or not row["label"]:
            continue

        text = row["review"].strip()
        label = int(row["label"])

        sample = {
            "text": text,
            "input_ids": encode_text(text),
            "label": label
        }

        if label == 1:
            positive_samples.append(sample)
        else:
            negative_samples.append(sample)


# --------------------------------------------------
# 5. 分别打乱正面和负面样本
# --------------------------------------------------

random.shuffle(positive_samples)
random.shuffle(negative_samples)


# --------------------------------------------------
# 6. 每一类分别按照 80% / 20% 划分
# --------------------------------------------------

def split_samples(samples):
    test_size = int(len(samples) * TEST_RATIO)

    test_samples = samples[:test_size]
    train_samples = samples[test_size:]

    return train_samples, test_samples


positive_train, positive_test = split_samples(positive_samples)
negative_train, negative_test = split_samples(negative_samples)

train_samples = positive_train + negative_train
test_samples = positive_test + negative_test

# 再次打乱，避免正面全部排在负面前面
random.shuffle(train_samples)
random.shuffle(test_samples)


# --------------------------------------------------
# 7. 输出统计结果
# --------------------------------------------------

print("\n[3] 原始数据")
print("正面：", len(positive_samples))
print("负面：", len(negative_samples))

print("\n[4] 训练集")
print("总数：", len(train_samples))
print("正面：", sum(x["label"] == 1 for x in train_samples))
print("负面：", sum(x["label"] == 0 for x in train_samples))

print("\n[5] 测试集")
print("总数：", len(test_samples))
print("正面：", sum(x["label"] == 1 for x in test_samples))
print("负面：", sum(x["label"] == 0 for x in test_samples))


# --------------------------------------------------
# 8. 查看一次文本编码结果
# --------------------------------------------------

example = test_samples[0]

print("\n[6] 编码示例")
print("原始文本：")
print(example["text"])

print("\n真实标签：", example["label"])

print("\n编码后的前30个数字：")
print(example["input_ids"][:30])

print("\n编码后总长度：", len(example["input_ids"]))


# --------------------------------------------------
# 9. 保存划分结果
# --------------------------------------------------

with open("data/train.json", "w", encoding="utf-8") as f:
    json.dump(train_samples, f, ensure_ascii=False)

with open("data/test.json", "w", encoding="utf-8") as f:
    json.dump(test_samples, f, ensure_ascii=False)

print("\n[7] 数据保存完成")
print("训练集：data/train.json")
print("测试集：data/test.json")

print("\n" + "=" * 60)
print("训练集与测试集准备成功！")
print("浙江工业大学 · 底气猫猫虫")
print("=" * 60)