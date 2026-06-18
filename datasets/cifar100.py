"""
=====================================================================
  CIFAR-100 数据集模块
=====================================================================

功能说明：
  本模块负责 CIFAR-100 数据集的加载、预处理和 few-shot 采样。
  CIFAR-100 是一个经典的细粒度图像分类数据集，包含 100 个类别。

数据集信息：
  - 训练集：50,000 张图片（每类 500 张）
  - 测试集：10,000 张图片（每类 100 张）
  - 图像尺寸：32 × 32 × 3（RGB 彩色）
  - 类别覆盖：动物（如 apple、bear、butterfly）、
              交通工具（如 bicycle、bus、train）、
              家具（如 bed、chair、table）、
              自然物（如 cloud、forest、mountain）等

模块功能：
  1. CIFAR100_CLASSES：100 个类别名称的完整列表
  2. get_default_transform()：CLIP 所需的图像预处理流程
  3. build_cifar100_fewshot()：构建 few-shot 训练集和测试集
  4. build_cifar100_fewshot_with_cache()：额外返回 cache 集（供扩展使用）

依赖：
  - torchvision：自动下载和管理 CIFAR-100 数据集
  - torch：数据集加载和 Subset 操作
"""

import torch
import torchvision          # PyTorch 视觉工具包，提供标准数据集和预处理
import torchvision.transforms as T  # 图像预处理变换
import numpy as np                    # 随机采样
from torch.utils.data import Dataset, Subset  # Subset 用于从数据集中提取子集


# =============================================================
# CIFAR-100 类别名称
# =============================================================
# 注意：这些类别名称与 torchvision 官方定义的类别顺序完全一致。
# 顺序对应关系：下标 0 → 'apple'，下标 1 → 'aquarium_fish'，...，下标 99 → 'worm'
# 这个对应关系至关重要，因为数据集的标签（整数索引）直接对应此列表中的位置。
CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver',       # 0-4
    'bed', 'bee', 'beetle', 'bicycle', 'bottle',              # 5-9
    'bowl', 'boy', 'bridge', 'bus', 'butterfly',              # 10-14
    'camel', 'can', 'castle', 'caterpillar', 'cattle',        # 15-19
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',     # 20-24
    'couch', 'crab', 'crocodile', 'cup', 'dinosaur',          # 25-29
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox',       # 30-34
    'girl', 'hamster', 'house', 'kangaroo', 'keyboard',       # 35-39
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',        # 40-44
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', # 45-49
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',      # 50-54
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',# 55-59
    'plain', 'plate', 'poppy', 'porcupine', 'possum',         # 60-64
    'rabbit', 'raccoon', 'ray', 'road', 'rocket',             # 65-69
    'rose', 'sea', 'seal', 'shark', 'shrew',                  # 70-74
    'skunk', 'skyscraper', 'snail', 'snake', 'spider',        # 75-79
    'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', # 80-84
    'tank', 'telephone', 'television', 'tiger', 'tractor',    # 85-89
    'train', 'trout', 'tulip', 'turtle', 'wardrobe',          # 90-94
    'whale', 'willow_tree', 'wolf', 'woman', 'worm',          # 95-99
]


def get_default_transform():
    """
    获取 CLIP 模型所需的默认图像预处理流程。

    为什么需要这个预处理？
      CLIP 在训练时使用的图像尺寸为 224×224，而 CIFAR-100 的原始
      图片尺寸只有 32×32。因此需要进行上采样（resize）和裁剪（crop）。

    预处理步骤（按顺序）：
      1. Resize(224, bicubic)
         - 将 32×32 放大到 224×224
         - 使用双三次插值（BICUBIC），质量最高但计算稍慢
         - CLIP 官方推荐使用 BICUBIC 插值

      2. CenterCrop(224)
         - 从中心裁剪 224×224 区域
         - 对 224×224 的图不会有实际裁剪效果
         - 保留是为了兼容输入尺寸大于 224 的情况

      3. ToTensor()
         - 将 PIL Image (H×W×C, 0-255) 转为 Tensor (C×H×W, 0.0-1.0)
         - 像素值从 [0, 255] 缩放到 [0.0, 1.0]

      4. Normalize(mean, std)
         - 使用 CLIP 训练时的 ImageNet 统计数据做标准化
         - output = (input - mean) / std
         - 经过标准化后，数据分布接近标准正态分布

    归一化参数来源（CLIP 官方）：
      mean = [0.48145466, 0.4578275, 0.40821073]   # RGB 三通道均值
      std  = [0.26862954, 0.26130258, 0.27577711]   # RGB 三通道标准差

    返回值：
        torchvision.transforms.Compose — 组合后的预处理流水线
    """
    return T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),  # 双三次插值放大到 224×224
        T.CenterCrop(224),                                          # 中心裁剪
        T.ToTensor(),                                               # PIL Image → Tensor
        T.Normalize(                                                # 标准化
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        ),
    ])


def build_cifar100_fewshot(shot, seed=42, transform=None, data_root='./datasets/cifar100'):
    """
    构建 CIFAR-100 的 few-shot 数据集。

    核心功能：
      从 CIFAR-100 的训练集（50000 张）中，为每个类别随机抽取
      shot 张图片，组成 few-shot 训练集。同时返回官方测试集
      （10000 张）用于评估。

    采样策略：
      - 每个类别独立采样，确保所有类别在训练集中都有 shot 张样本
      - 采样是类别均衡的（class-balanced），避免类别不平衡问题
      - 使用 numpy.random.RandomState(seed) 保证可重复性

    参数详解：
        shot      : int  — 每个类别使用的训练样本数
                           支持的值：1, 2, 4, 8, 16（论文标准设置）
                           任意正整数也可，但至少为 1
        seed      : int  — 随机种子，固定为 42 确保可重复性
                           改变种子会导致不同的采样结果
        transform : callable or None — 图像预处理函数
                           None 时使用 get_default_transform()
        data_root : str  — 数据集在磁盘上的根目录
                           如果数据集不存在，会自动从网络下载

    返回值（tuple）：
        train_set : torch.utils.data.Subset
            few-shot 训练集，包含 shot × 100 张图片
            数据量示例：1-shot → 100 张，16-shot → 1600 张
        test_set  : torchvision.datasets.CIFAR100
            官方测试集，包含 10000 张图片，每类 100 张
            注意：测试集在所有 shot 设置下完全相同

    使用示例：
        # 构建 4-shot 数据集
        train_set, test_set = build_cifar100_fewproof(shot=4)
        print(len(train_set))  # 输出: 400

        # 构建 16-shot 数据集
        train_set, test_set = build_cifar100_fewproof(shot=16)
        print(len(train_set))  # 输出: 1600
    """
    if transform is None:
        transform = get_default_transform()  # 使用默认的 CLIP 预处理

    # ===== 加载完整训练集 =====
    # CIFAR-100 训练集包含 50000 张图片（每类 500 张）
    # download=True：如果本地不存在，自动从网络下载
    full_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )

    # ===== 加载测试集 =====
    # CIFAR-100 测试集包含 10000 张图片（每类 100 张）
    # 测试集在所有实验中保持一致，不进行 few-shot 采样
    test_set = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform
    )

    # ===== Few-shot 采样 =====
    # 从每类的 500 张训练图片中随机抽取 shot 张
    rng = np.random.RandomState(seed)  # 可控随机数生成器
    targets = np.array(full_train.targets)  # 所有训练图片的标签
    num_classes = 100                    # CIFAR-100 有 100 个类别
    selected_indices = []                # 存放抽取的样本索引

    # 对每个类别分别采样
    for c in range(num_classes):
        # 找到当前类别 c 的所有样本索引
        idxs = np.where(targets == c)[0]

        # 从该类中随机抽取 shot 张（不重复采样）
        # min(shot, len(idxs)) 防止 shot 超过该类的总样本数（500）
        chosen = rng.choice(idxs, size=min(shot, len(idxs)), replace=False)
        selected_indices.extend(chosen.tolist())

    # 使用 Subset 从完整训练集中提取采样的子集
    # Subset 是 PyTorch 的轻量级数据集包装器，只保存索引而不复制数据
    train_set = Subset(full_train, selected_indices)

    return train_set, test_set


def build_cifar100_fewshot_with_cache(shot, seed=42, transform=None, data_root='./datasets/cifar100'):
    """
    与 build_cifar100_fewshot 功能相同，但额外返回 cache_set。

    用途说明：
      此函数专为 Tip-Adapter（一种基于 cache 模型的 few-shot 方法）
      等需要额外特征缓存的方法设计。后续扩展时可基于此函数实现。

    额外返回值：
        cache_set : torch.utils.data.Subset
            与 train_set 使用相同的采样策略和种子，
            可用作特征缓存的构建数据集
            内容与 train_set 完全相同，只是独立的对象引用

    参数说明（同 build_cifar100_fewshot）：
        shot, seed, transform, data_root

    返回值（tuple of 3）：
        train_set, test_set, cache_set
    """
    if transform is None:
        transform = get_default_transform()

    # 调用基础函数获取训练集和测试集
    train_set, test_set = build_cifar100_fewshot(shot, seed, transform, data_root)

    # 重新加载完整训练集（因为 Subset 不保留原始数据集的完整引用）
    full_train = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform
    )

    # 使用与 build_cifar100_fewshot 完全相同的采样逻辑
    rng = np.random.RandomState(seed)
    targets = np.array(full_train.targets)
    num_classes = 100
    selected_indices = []

    for c in range(num_classes):
        idxs = np.where(targets == c)[0]
        chosen = rng.choice(idxs, size=min(shot, len(idxs)), replace=False)
        selected_indices.extend(chosen.tolist())

    # 创建独立的 cache_set（虽然内容相同，但对象不同）
    cache_set = Subset(full_train, selected_indices)

    return train_set, test_set, cache_set
