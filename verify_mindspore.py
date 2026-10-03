import mindspore as ms
from mindspore import Tensor
import numpy as np

print("=" * 60)
print("浙江工业大学 · 底气猫猫虫")
print("MindSpore 中文文本分析自主学习实验")
print("=" * 60)

print("\n[1] MindSpore版本")
print("MindSpore Version:", ms.__version__)

print("\n[2] 运行设备")
print("Device Target:", ms.get_context("device_target"))

print("\n[3] Tensor创建测试")
x = Tensor(np.array([1, 2, 3]), ms.float32)
y = Tensor(np.array([10, 20, 30]), ms.float32)

print("x =", x)
print("y =", y)

print("\n[4] Tensor计算测试")
result = x + y
print("x + y =", result)

print("\n" + "=" * 60)
print("MindSpore 环境配置与计算测试成功！")
print("浙江工业大学 · 底气猫猫虫")
print("=" * 60)