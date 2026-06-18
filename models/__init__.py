"""
models 包 —— 三种 CLIP few-shot 分类模型的统一导出入口。

使用方式：
    from models import CLIPZeroShot, CoOp, CLIPAdapter

支持的模型：
    - CLIPZeroShot : CLIP 零样本分类（基线方法，无需训练）
    - CoOp         : Context Optimization，可学习 prompt 向量
    - CLIPAdapter  : CLIP-Adapter，特征空间适配微调
"""
from .clip_zeroshot import CLIPZeroShot
from .coop import CoOp
from .clip_adapter import CLIPAdapter
