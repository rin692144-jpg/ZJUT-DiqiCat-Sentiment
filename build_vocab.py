import csv
import json
from collections import Counter
from pathlib import Path

print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("中文文本字符词表构建")
print("=" * 60)

data_file = Path("data") / "ChnSentiCorp_htl_all.csv"
vocab_file = Path("data") / "vocab.json"

reviews = []

# 读取全部评论
with open(data_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["review"] and row["label"]:
            reviews.append(row["review"].strip())

print("\n[1] 有效评论数：", len(reviews))

# 统计每个字符出现的次数
char_counter = Counter()

for review in reviews:
    char_counter.update(review)

print("[2] 不重复字符数：", len(char_counter))

# 只保留至少出现 2 次的字符
min_frequency = 2

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

# 出现次数高的字符优先加入词表
for char, count in char_counter.most_common():
    if count >= min_frequency:
        vocab[char] = len(vocab)

print("[3] 最低字符频率：", min_frequency)
print("[4] 最终词表大小：", len(vocab))

# 保存词表
with open(vocab_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("[5] 词表保存位置：", vocab_file)

# 演示文字如何转换成数字
example = "浙江工业大学底气猫猫虫真不错"

encoded = [
    vocab.get(char, vocab["<UNK>"])
    for char in example
]

print("\n[6] 文本编码演示")
print("原始文本：", example)
print("字符序列：", list(example))
print("数字序列：", encoded)

print("\n[7] 高频字符 Top 20")

for char, count in char_counter.most_common(20):
    print(repr(char), "→", count)

print("\n" + "=" * 60)
print("字符词表构建完成！")
print("浙江工业大学 · 底气猫猫虫")
print("=" * 60)