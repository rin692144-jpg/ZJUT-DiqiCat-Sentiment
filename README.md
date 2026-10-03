# 基于 MindSpore 的中文酒店评论情感分类

> 浙江工业大学 · 底气猫猫虫 · 文本分析自主学习实验

以国产深度学习框架 **MindSpore 2.10.0** 为平台，在 Windows CPU 环境下完成中文文本情感二分类的完整流程：环境部署 → 数据分析 → 字符词表 → 文本编码 → 训练/测试集划分 → 网络搭建 → 模型训练 → 指标评价 → 自定义文本推理。

## 实验环境

| 项目 | 版本 |
| --- | --- |
| 操作系统 | Windows |
| Python | 3.12.10（MindSpore 无 3.13 的 Windows 安装包） |
| MindSpore | 2.10.0（CPU） |
| NumPy | 1.26.4 |
| SciPy | 1.14.1 |

## 快速开始

```bash
# 1. 创建虚拟环境并安装依赖
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. 按顺序运行
python verify_mindspore.py    # 验证 MindSpore 环境
python prepare_data.py        # 下载 ChnSentiCorp 数据集
python analyze_data.py        # 数据统计分析
python build_vocab.py         # 构建字符级词表
python prepare_dataset.py     # 生成 train.json / test.json
python train.py               # 训练并保存最佳模型
python evaluate.py            # 混淆矩阵与评价指标
python predict.py             # 自定义文本预测
```

## 目录结构

```
.
├── data/                     # 数据集与词表
│   ├── ChnSentiCorp_htl_all.csv
│   ├── vocab.json            # 字符级词表（build_vocab.py 生成）
│   ├── train.json            # 训练集（prepare_dataset.py 生成）
│   └── test.json             # 测试集（prepare_dataset.py 生成）
├── models/
│   └── sentiment_model.ckpt  # 最佳模型参数
├── verify_mindspore.py
├── prepare_data.py
├── analyze_data.py
├── build_vocab.py
├── prepare_dataset.py
├── train.py
├── evaluate.py
└── predict.py
```

## 模型结构

字符级 `Embedding(3115, 64)` → Mask 平均池化 → `Dense(64, 2)`，损失函数 CrossEntropyLoss，优化器 Adam（lr=0.001），Batch Size 64，训练 10 个 Epoch。

## 实验结果

- 有效样本 7765 条（正面 5322 / 负面 2443），按 8∶2 分层划分。
- 最佳测试准确率 **87.44%**（第 9 个 Epoch）。
- Positive Precision 89.18%、Recall 92.95%、F1 91.03%；Negative Recall 75.41%。

## 原创性说明

训练脚本、预测输出、测试输入中均包含“浙江工业大学底气猫猫虫”标识，用于确认实验由本组独立完成。

## 说明

- 本仓库不含 `.venv/` 虚拟环境（体积过大，请按上方命令自行安装）。
- 词表与训练/测试集为脚本生成产物，也可由 `build_vocab.py`、`prepare_dataset.py` 重新生成。
