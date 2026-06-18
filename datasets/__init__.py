"""
datasets 包 —— 数据集加载与 few-shot 采样模块的统一导出入口。

使用方式：
    from datasets import build_cifar100_fewshot, CIFAR100_CLASSES

导出内容：
    - build_cifar100_fewshot(shot, seed, transform, data_root)
        构建 few-shot 训练集和完整测试集
    - CIFAR100_CLASSES
        CIFAR-100 的 100 个类别名称列表
"""
from .cifar100 import build_cifar100_fewshot, CIFAR100_CLASSES
