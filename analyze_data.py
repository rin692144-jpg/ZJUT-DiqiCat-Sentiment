import csv
from pathlib import Path

print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("ChnSentiCorp 中文情感数据集分析")
print("=" * 60)

data_file = Path("data") / "ChnSentiCorp_htl_all.csv"

reviews = []
labels = []

# utf-8-sig 可以兼容 CSV 文件开头可能存在的 BOM
with open(data_file, "r", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)

    for row in reader:
        # 跳过空评论
        if row["review"] and row["label"]:
            reviews.append(row["review"].strip())
            labels.append(int(row["label"]))

positive_count = labels.count(1)
negative_count = labels.count(0)

print("\n[1] 数据总量")
print("评论总数：", len(reviews))

print("\n[2] 情感类别统计")
print("正面评论：", positive_count)
print("负面评论：", negative_count)

print("\n[3] 类别比例")
print("正面比例：{:.2f}%".format(positive_count / len(labels) * 100))
print("负面比例：{:.2f}%".format(negative_count / len(labels) * 100))

print("\n[4] 随机查看前5条评论")

for i in range(min(5, len(reviews))):
    sentiment = "正面" if labels[i] == 1 else "负面"

    print("\n第 {} 条".format(i + 1))
    print("标签：{} ({})".format(labels[i], sentiment))
    print("评论：", reviews[i])

print("\n" + "=" * 60)
print("数据集分析完成！")
print("浙江工业大学 · 底气猫猫虫")
print("=" * 60)