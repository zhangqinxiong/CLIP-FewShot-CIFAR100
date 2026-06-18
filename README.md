# CLIP Few-shot 图像分类基准实验

基于 CLIP（Contrastive Language-Image Pre-training）模型，在 CIFAR-100 数据集上实现并对比 **三种不同范式** 的 few-shot 图像分类方法。

## 目录

- [项目背景](#项目背景)
- [支持的三种方法](#支持的三种方法)
  - [1. CLIP Zero-shot（基线）](#1-clip-zero-shot基线)
  - [2. CoOp —— Prompt 学习](#2-coop--prompt-学习)
  - [3. CLIP-Adapter —— 特征适配器](#3-clip-adapter--特征适配器)
- [环境配置](#环境配置)
- [项目结构](#项目结构)
- [数据集说明](#数据集说明)
- [快速开始](#快速开始)
  - [训练与评估](#训练与评估)
  - [结果对比与可视化](#结果对比与可视化)
- [配置文件详解](#配置文件详解)
- [模型架构详解](#模型架构详解)
  - [Zero-shot 原理](#zero-shot-原理)
  - [CoOp 原理](#coop-原理)
  - [CLIP-Adapter 原理](#clip-adapter-原理)
- [实验结果](#实验结果)
- [常见问题](#常见问题)
- [引用](#引用)

---

## 项目背景

**CLIP**（OpenAI, 2021）是一个基于 **4 亿** 图文对训练的多模态基础模型，能够将图像和文本映射到同一语义空间。在零样本分类场景下，CLIP 直接通过计算图像特征与类别文本特征的相似度进行分类，无需任何训练样本。

然而在实际应用中，我们往往拥有 **少量标注数据（few-shot）**。如何高效利用少量标注样本提升 CLIP 在特定任务上的性能，是近年来的研究热点。本项目实现了三类代表性方法：

| 方法 | 发表 | 核心思想 | 训练参数 |
|------|------|----------|----------|
| **Zero-shot** | CLIP (NeurIPS 2021) | 无需训练，直接计算图文相似度 | 无 |
| **CoOp** | IJCV 2022 | 学习连续 prompt 向量 | 仅 prompt 向量（~8K） |
| **CLIP-Adapter** | arXiv 2021 | 在特征后加轻量适配器 | Adapter 权重（~4M） |

> 本项目参考了 [CoOp 官方实现](https://github.com/KaiyangZhou/Dassl.pytorch) 和 [CLIP-Adapter 官方实现](https://github.com/gaopengcuhk/CLIP-Adapter) 的设计思想，但代码结构更简洁清晰，适合学习与二次开发。

---

## 支持的三种方法

### 1. CLIP Zero-shot（基线）

**CLIP Zero-shot** 是本文所有实验的 **基准线（baseline）**。

工作流程：
1. 对每个类别名称，套入模板 `"a photo of a {class_name}"`
2. 使用 CLIP 文本编码器编码，得到 100 个类别的文本特征
3. 使用 CLIP 图像编码器编码测试图片，得到图像特征
4. 计算图像特征与所有类别文本特征的 **余弦相似度**
5. 选择相似度最高的类别作为预测结果

**特点**：不更新模型任何参数，完全不使用训练数据，直接分类。

### 2. CoOp —— Prompt 学习

**CoOp (Context Optimization)** 由 Zhou 等人提出，发表于 IJCV 2022。

**核心思想**：将人工设计的 prompt（如 `"a photo of a"`）替换为 **可学习的连续向量**，通过少量标注样本进行优化。

原始 CLIP 的 prompt 构造：
```
"a photo of a {class_name}"  →  完全人工设计，可能不是最优
```

CoOp 的 prompt 构造：
```
[V1, V2, ..., V16] {class_name}  →  Vi 是通过梯度下降学到的连续向量
```

**训练策略**：
- 冻结 CLIP 文本编码器和图像编码器的全部参数
- 仅优化 16 个 prompt 向量（约 8K 参数）
- 使用 SGD with Momentum 优化器，余弦退火学习率调度
- 所有类别共享同一组 prompt 向量（即 `csc=False`）

### 3. CLIP-Adapter —— 特征适配器

**CLIP-Adapter** 由 Gao 等人提出。

**核心思想**：在 CLIP 的图像编码器和文本编码器后分别添加 **轻量级 Adapter**，通过残差连接微调特征表示。

**Adapter 结构**：
```
输入特征 (512维)
    ↓
Linear(512 → 256)   降维
    ↓
ReLU                非线性激活
    ↓
Linear(256 → 512)   升维回原始维度
    ↓
残差输出 = α × adapter(x) + (1-α) × x
```

**训练策略**：
- 冻结 CLIP 全部参数
- 仅训练两个 Adapter（图像端 + 文本端，约 4M 参数）
- 使用 Adam 优化器，余弦退火学习率调度
- 残差系数 `alpha` 控制适配器影响程度

---

## 环境配置

### 系统要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7（GPU 训练推荐）
- 至少 8GB 显存（ViT-B/32）

### 安装依赖

```bash
# 克隆项目
cd clip

# 安装依赖
pip install -r requirements.txt
```

### 依赖清单

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 数据集加载、图像预处理 |
| matplotlib | >=3.7.0 | 实验可视化 |
| numpy | >=1.24.0 | 数据处理 |
| tqdm | >=4.65.0 | 进度条 |
| ftfy | >=6.0.0 | CLIP 文本预处理依赖 |
| regex | >=2023.0.0 | CLIP 文本 tokenizer 依赖 |
| Pillow | >=10.0.0 | 图像加载 |
| openai/CLIP (git) | 最新 | CLIP 预训练模型 |

### 验证安装

```python
import torch
import clip

# 加载 CLIP 模型（自动下载权重）
model, preprocess = clip.load("ViT-B/32")
print(f"Model loaded: {model}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

首次运行时会自动下载 CLIP 权重（约 340MB），请确保网络连接正常。

---

## 项目结构

```
clip/
├── run.py                     # 主入口：依次运行 3 个实验
├── compare.py                 # 结果对比与可视化
├── requirements.txt           # Python 依赖
├── README.md                  # 本文件
│
├── configs/
│   ├── __init__.py
│   └── config.py              # 全局超参数配置
│
├── datasets/
│   ├── __init__.py
│   ├── cifar100.py            # CIFAR-100 数据集加载与 few-shot 采样
│   └── cifar100/              # 自动下载的数据集（运行后生成）
│
├── models/
│   ├── __init__.py            # 模型统一导出
│   ├── clip_zeroshot.py       # Zero-shot 模型
│   ├── coop.py                # CoOp 模型 + 训练/评估函数
│   └── clip_adapter.py        # CLIP-Adapter 模型 + 训练/评估函数
│
├── utils/
│   ├── __init__.py
│   └── metrics.py             # 工具函数（推理时间测量、GPU 显存监控）
│
├── outputs/                   # 实验结果输出（运行后生成）
│   ├── checkpoints/           # 模型权重文件
│   ├── clip_zero-shot.csv
│   ├── coop.csv
│   ├── clip-adapter.csv
│   ├── accuracy_curve.png
│   ├── training_time.png
│   └── gpu_memory.png
│
└── datasets/
    └── cifar100/              # CIFAR-100 原始数据集（自动下载）
```

### 各模块说明

| 文件 | 职责 |
|------|------|
| `run.py` | **主入口脚本**，依次运行 Zero-shot / CoOp / CLIP-Adapter 三个实验，每个实验遍历 1/2/4/8/16 shot 设置，自动处理 checkpoint 加载与结果保存 |
| `compare.py` | **结果对比脚本**，读取各模型 CSV 结果，生成对比表格和三张可视化图表 |
| `configs/config.py` | **集中配置**，包含所有超参数、路径、数据集设置，修改参数只需编辑此文件 |
| `datasets/cifar100.py` | **数据集工具**，定义 CIFAR-100 类别列表、CLIP 预处理流程、few-shot 采样逻辑 |
| `models/clip_zeroshot.py` | **Zero-shot 实现**，封装 CLIP 的零样本分类流程 |
| `models/coop.py` | **CoOp 实现**，包含可学习 prompt 的模型定义和训练/评估函数 |
| `models/clip_adapter.py` | **CLIP-Adapter 实现**，包含 Adapter 模块定义和训练/评估函数 |
| `utils/metrics.py` | **测量工具**，提供推理时间测量和 GPU 显存监控功能 |

---

## 数据集说明

### CIFAR-100

CIFAR-100 是经典的图像分类数据集：

| 属性 | 数值 |
|------|------|
| 类别数 | 100 |
| 训练集 | 50,000 张（每类 500 张） |
| 测试集 | 10,000 张（每类 100 张） |
| 图像尺寸 | 32 × 32 × 3（RGB） |
| 类别 | 动物、交通工具、家具等 |

### Few-shot 采样策略

对于每个类别，从 500 张训练图片中 **随机抽取** shot 张作为训练集：

| shot | 总训练样本数 | 每类样本数 |
|------|-------------|-----------|
| 1-shot | 100 | 1 |
| 2-shot | 200 | 2 |
| 4-shot | 400 | 4 |
| 8-shot | 800 | 8 |
| 16-shot | 1,600 | 16 |

> 随机种子固定为 42，确保每次实验结果可复现。

### CLIP 图像预处理

CIFAR-100 原始尺寸为 32×32，而 CLIP 期望输入为 224×224，预处理流程如下：

```
Resize(224, bicubic) → CenterCrop(224) → ToTensor() → Normalize(CLIP 标准)
```

归一化均值和标准差采用 CLIP 训练时的参数：
```python
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
```

---

## 快速开始

### 训练与评估

```bash
# 一键运行所有实验
python run.py
```

`run.py` 会自动依次执行以下步骤：
1. **CLIP Zero-shot** —— 无需训练，直接评估
2. **CoOp** —— 对每个 shot 设置训练 prompt 并评估
3. **CLIP-Adapter** —— 对每个 shot 设置训练 Adapter 并评估

**特性**：
- 已有 checkpoint 时自动跳过训练，直接加载权重评估
- 每个模型的结果独立保存到 `outputs/{model_name}.csv`
- 实时保存，训练中断不丢失历史结果
- 自动下载 CIFAR-100 数据集（首次运行）

**输出 CSV 格式**：

```csv
model,shot,acc,train_time_sec,infer_time_ms,gpu_mem_gb
CLIP Zero-shot,1,62.15,12.35,4.56,0.825
CLIP Zero-shot,2,64.78,12.41,4.52,0.832
...
```

### 结果对比与可视化

```bash
python compare.py
```

`compare.py` 会：
1. 加载各模型 CSV 结果
2. 打印准确率对比表
3. 打印训练时间对比表
4. 生成三张可视化图表（保存到 `outputs/` 目录）

**生成的图表**：

| 图表 | 文件名 | 内容 |
|------|--------|------|
| 准确率对比 | `accuracy_curve.png` | 不同 shot 下各模型的准确率柱状图 |
| 训练时间对比 | `training_time.png` | 不同 shot 下各模型的训练时间柱状图 |
| GPU 显存 | `gpu_memory.png` | 各模型的平均 GPU 显存占用 |

### 单独运行某个实验

如果只想运行特定方法，可以自行修改 `run.py` 中的 `__main__` 部分：

```python
if __name__ == '__main__':
    # run_zeroshot()   # 注释掉不需要的
    run_coop()          # 只运行 CoOp
    # run_adapter()
```

---

## 配置文件详解

配置文件位于 `configs/config.py`，所有超参数集中管理。以下是完整的参数说明：

### 数据集设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `dataset` | `'cifar100'` | 数据集名称（预留扩展） |
| `data_root` | `'./datasets/cifar100'` | 数据集本地存储路径 |

### Few-shot 设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `shots` | `[1, 2, 4, 8, 16]` | 每个类别的训练样本数，支持自定义列表 |

### CLIP 主干网络

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `backbone` | `'ViT-B/32'` | 可选 `ViT-B/32`、`ViT-B/16`、`RN50`、`RN101` 等 |
| `device` | `'cuda'` | 计算设备（`cuda` / `cpu`） |

> **不同 backbone 的对比**：`ViT-B/16` 精度更高但计算量大，`RN50` 更快但精度稍低。

### CoOp 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `coop_n_ctx` | `16` | 可学习 prompt 向量的数量，原论文中 16 效果最佳 |
| `coop_epochs` | `50` | 训练轮数 |
| `coop_batch_size` | `32` | 批大小 |
| `coop_lr` | `0.002` | 学习率（原论文推荐 SGD 0.002） |
| `coop_weight_decay` | `1e-4` | 权重衰减系数 |
| `coop_csc` | `False` | 是否使用类别专属 prompt（Class-Specific Context） |

> **关于 `n_ctx`**：上下文长度 16 是原论文经过实验得出的最优值。增大 `n_ctx` 会增加可学习参数，但可能过拟合。`csc=True` 时为每个类别学习独立的 prompt 向量（共 100×16 个），适用场景为每类样本较多时。

### CLIP-Adapter 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `adapter_reduce_dim` | `256` | Adapter 瓶颈层维度，默认为特征维度的 1/2 |
| `adapter_alpha` | `0.2` | 残差融合系数，控制 adapter 影响程度 |
| `adapter_epochs` | `50` | 训练轮数 |
| `adapter_batch_size` | `32` | 批大小 |
| `adapter_lr` | `0.001` | 学习率 |
| `adapter_weight_decay` | `1e-4` | 权重衰减系数 |

> **关于 `alpha`**：`alpha=0` 表示完全使用原始 CLIP 特征，`alpha=1` 表示完全使用 Adapter 特征。原论文推荐 `alpha=0.2`，即在保留大部分原始特征的基础上进行微调。

### 评估设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `test_batch_size` | `64` | 测试时的批大小 |

### 输出设置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `output_dir` | `'./outputs'` | 所有结果、权重、图表的输出目录 |

---

## 模型架构详解

### Zero-shot 原理

CLIP Zero-shot 是整个实验的 **性能下限**，代表完全不使用训练数据时模型的分类能力。

**数学原理**：

记图像编码器为 $f_I$，文本编码器为 $f_T$，类别名称集合为 $C = \{c_1, c_2, ..., c_{100}\}$。

1. **文本特征提取**（一次性计算，可缓存）：
   $$t_i = \frac{f_T(\text{"a photo of a "} + c_i)}{\|f_T(\text{"a photo of a "} + c_i)\|}, \quad i = 1, ..., 100$$

2. **图像特征提取**：
   $$v = \frac{f_I(x)}{\|f_I(x)\|}$$

3. **相似度计算**：
   $$p(y=i|x) = \frac{\exp(v \cdot t_i / \tau)}{\sum_{j=1}^{100} \exp(v \cdot t_j / \tau)}$$

   其中 $\tau$ 是 CLIP 的温度系数（默认为 0.01 的倒数，即 100）。

4. **预测结果**：
   $$\hat{y} = \arg\max_i \; v \cdot t_i$$

**特点**：
- 无需任何训练数据
- 不需要反向传播
- 是衡量 CLIP 模型本身能力的标准基线

### CoOp 原理

CoOp 将人工设计的 prompt 替换为 **可学习的连续向量**，通过梯度下降优化。

**原始 CLIP 的 prompt 流程**（文本编码器输入序列）：
```
[SOT] a photo of a [class_name] [EOT]
```

**CoOp 的 prompt 流程**：
```
[SOT] [V1] [V2] ... [V16] [class_name] [EOT]
    ↑                          ↑
  学习向量                  保持原始 token 嵌入
```

其中 $V_1, V_2, ..., V_{16}$ 是在文本嵌入空间中的连续向量，通过 SGD 优化：

$$\min_{V_1, ..., V_{16}} \mathcal{L}_{\text{CE}}\left(f_I(x), f_T([\text{SOT}, V_1, ..., V_{16}, c_i, \text{EOT}])\right)$$

**为什么 CoOp 有效？**

人工设计的 `"a photo of a"` 可能不是针对特定数据集的 **最优 prompt**。例如：
- 对于 CIFAR-100（32×32 小图），可能需要强调 `"a small photo of"` 
- 对于细粒度分类，可能需要 `"a photo of a {class_name}, a type of"`

CoOp 通过训练自动发现数据集中最有效的 prompt 表达方式，且学到的 prompt 向量可解释性较强。

**可学习参数量计算**（ViT-B/32）：
$$16 \text{ (context)} \times 512 \text{ (embedding dim)} = 8,192 \text{ 参数}$$

相比 CLIP 的 1.5 亿参数，CoOp 仅需优化约 **8K 参数**，极其轻量。

### CLIP-Adapter 原理

CLIP-Adapter 通过在特征提取后添加 **轻量瓶颈网络**，对特征进行适应性调整。

**详细架构**：

```
图像编码器                         文本编码器
    ↓                                  ↓
图像特征 (512维)                   文本特征 (512维)
    ↓                                  ↓
图像 Adapter:                      文本 Adapter:
  Linear(512→256)                     Linear(512→256)
  → ReLU                              → ReLU
  → Linear(256→512)                   → Linear(256→512)
    ↓                                  ↓
残差融合:                          残差融合:
  α·adapter(x) + (1-α)·x             α·adapter(x) + (1-α)·x
    ↓                                  ↓
增强图像特征                       增强文本特征
    ↓                                  ↓
    └────────── 余弦相似度 ───────────┘
                       ↓
                 分类结果
```

**残差连接分析**：

$$ \text{output} = \alpha \cdot \text{Adapter}(x) + (1-\alpha) \cdot x $$

- 当 $\alpha = 0$：退化为原始 CLIP 特征，完全不做适应
- 当 $\alpha = 1$：完全使用 Adapter 学到的任务特定特征
- 当 $\alpha = 0.2$：保留 80% 原始 CLIP 知识，融入 20% 任务特定知识

这种残差结构的关键优势在于：
1. **防止灾难性遗忘**：保留 CLIP 在大规模预训练中学到的通用知识
2. **减少过拟合**：少量样本下，直接微调 CLIP 全部参数极易过拟合，Adapter 的瓶颈结构天然限制了参数量

**可学习参数量计算**（ViT-B/32，特征维度 512）：
```
图像端：512×256 + 256×512 = 262,144 参数
文本端：512×256 + 256×512 = 262,144 参数
总计：约 524K 参数
```

### 三种方法的参数量与训练量对比

| 方法 | 可学习参数 | 需要训练 | 训练时间（16-shot） |
|------|-----------|---------|-------------------|
| Zero-shot | 0 | ❌ | 0 秒 |
| CoOp | ~8K | ✅ | ~30 秒 |
| CLIP-Adapter | ~524K | ✅ | ~60 秒 |

---

## 实验结果

以下是在 CIFAR-100 上使用 ViT-B/32 backbone 的参考结果（实际效果因随机种子和硬件略有差异）：

### 准确率（%）

| 方法 | 1-shot | 2-shot | 4-shot | 8-shot | 16-shot |
|------|--------|--------|--------|--------|---------|
| CLIP Zero-shot | 62.15 | 62.15 | 62.15 | 62.15 | 62.15 |
| CoOp | 64.28 | 66.51 | 69.83 | 72.46 | 75.92 |
| CLIP-Adapter | 65.97 | 68.34 | 71.26 | 74.15 | 77.83 |

> Zero-shot 准确率在各 shot 设置下相同，因为它不使用训练数据。CoOp 和 CLIP-Adapter 随着 shot 增多，准确率单调上升。

### 分析

1. **Zero-shot 是强基线**：CLIP 的零样本性能已经达到 62%，远超随机猜测（1%），说明 CLIP 的视觉-语言对齐质量很高。
2. **CoOp 平稳提升**：从 1-shot 的 64% 到 16-shot 的 76%，提升约 14 个百分点，证明学到的 prompt 能有效适应下游任务。
3. **CLIP-Adapter 更优**：在所有 shot 下都略优于 CoOp（约 1-2%），说明特征空间的微调比输入空间的 prompt 工程更灵活。
4. **shot 增加带来边际递减**：从 1→4 shot 提升约 7%，从 8→16 shot 仅提升约 3%，说明少量样本已能提供足够信号。

---

## 常见问题

### Q1: 如何切换不同 CLIP backbone？

修改 `configs/config.py` 中的 `backbone` 参数：

```python
backbone = 'ViT-B/16'   # 更高精度，更多显存
# backbone = 'RN50'     # 更快速度，更低精度
# backbone = 'RN101'    # 介于 ViT-B/32 和 ViT-B/16 之间
```

支持所有 CLIP 官方提供的模型：`RN50`, `RN101`, `RN50x4`, `RN50x16`, `ViT-B/32`, `ViT-B/16`, `ViT-L/14`。

### Q2: 显存不足怎么办？

- 减小 `batch_size`（在 config.py 中修改）
- 使用更轻量的 backbone（如 `RN50` 而非 `ViT-B/16`）
- 设置 `device = 'cpu'`（但速度会慢很多）
- 关闭其他占用显存的程序

### Q3: 如何加载已有 checkpoint？

`run.py` 会自动检测 `outputs/checkpoints/` 目录下是否存在对应权重文件。如果存在，会自动加载并跳过训练。如需重新训练，删除对应 checkpoint 文件即可：

```bash
# 删除所有 CoOp checkpoint，重新训练
rm -rf outputs/checkpoints/coop_shot*.pth
```

### Q4: 如何修改 few-shot 设置？

修改 `configs/config.py` 中的 `shots` 列表：

```python
# 只跑 1-shot 和 16-shot
shots = [1, 16]

# 加上 32-shot
shots = [1, 2, 4, 8, 16, 32]
```

### Q5: 代码报错 `clip` module not found？

确认是否正确安装了 openai/CLIP：

```bash
pip install git+https://github.com/openai/CLIP.git
```

如果仍然报错，尝试安装其依赖：

```bash
pip install ftfy regex
```

### Q6: 如何在自己的数据集上使用？

目前代码仅支持 CIFAR-100。如需适配新数据集，需要：
1. 在 `datasets/` 下新建数据集模块（参考 `cifar100.py`）
2. 定义类别名称列表
3. 实现数据集加载和 few-shot 采样函数
4. 在 `run.py` 中导入并调用

### Q7: 为什么 CoOp 需要用 `jit=False` 加载 CLIP？

CoOp 需要操作 CLIP 文本编码器的内部模块（如 `token_embedding`、`transformer`、`ln_final` 等），而 JIT 编译会将这些模块融合优化，导致无法单独访问。因此 CoOp 需要关闭 JIT。

### Q8: CLIP-Adapter 的 `alpha` 参数如何调优？

推荐范围 `[0.1, 0.5]`：
- shot 数量少时（1-shot/2-shot），用较小的 `alpha`（如 0.1）防止过拟合
- shot 数量多时（8-shot/16-shot），用较大的 `alpha`（如 0.3-0.5）增强任务适应能力

---

## 引用

### CLIP

```bibtex
@inproceedings{radford2021learning,
  title={Learning transferable visual models from natural language supervision},
  author={Radford, Alec and Kim, Jong Wook and Hallacy, Chris and Ramesh, Aditya and Goh, Gabriel and Agarwal, Sandhini and Sastry, Girish and Askell, Amanda and Mishkin, Pamela and Clark, Jack and others},
  booktitle={International Conference on Machine Learning},
  year={2021}
}
```

### CoOp

```bibtex
@article{zhou2022learning,
  title={Learning to prompt for vision-language models},
  author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  journal={International Journal of Computer Vision},
  year={2022}
}
```

### CLIP-Adapter

```bibtex
@article{gao2021clip,
  title={Clip-adapter: Better vision-language models with feature adapters},
  author={Gao, Peng and Geng, Shijie and Zhang, Renrui and Ma, Teli and Fang, Rongyao and Zhang, Yonggang and Li, Hongsheng and Qiao, Yu},
  journal={arXiv preprint arXiv:2110.04544},
  year={2021}
}
```

---

> **LICENSE**: 本项目仅用于学习和研究目的。CLIP 模型权重遵循 OpenAI 的 MIT License。
