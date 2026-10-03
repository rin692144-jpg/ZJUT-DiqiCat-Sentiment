from pathlib import Path
import urllib.request

print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("中文情感分类实验——数据集准备")
print("=" * 60)

# 项目中的 data 文件夹
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# 数据保存位置
data_file = data_dir / "ChnSentiCorp_htl_all.csv"

# ChnSentiCorp 数据集
url = (
    "https://raw.githubusercontent.com/"
    "SophonPlus/ChineseNlpCorpus/master/"
    "datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
)

print("\n[1] 数据集：ChnSentiCorp_htl_all")
print("[2] 任务类型：中文酒店评论情感二分类")
print("[3] 标签含义：1 = 正面，0 = 负面")

if data_file.exists():
    print("\n数据集已经存在，无需重复下载。")
else:
    print("\n正在下载数据集...")
    urllib.request.urlretrieve(url, data_file)
    print("下载完成！")

print("\n[4] 文件位置：", data_file)
print("[5] 文件大小：{:.2f} MB".format(
    data_file.stat().st_size / 1024 / 1024
))

print("\n" + "=" * 60)
print("数据集准备完成！")
print("浙江工业大学 · 底气猫猫虫")
print("=" * 60)