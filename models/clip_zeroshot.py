"""
=====================================================================
  CLIP Zero-shot 零样本分类模型
=====================================================================

论文来源：
  Learning Transferable Visual Models From Natural Language Supervision
  (Radford et al., NeurIPS 2021)

核心思想：
  CLIP（Contrastive Language-Image Pre-training）通过 4 亿图文对
  的对比学习训练，将图像和文本映射到同一个语义空间。在零样本分类
  场景下，不需要任何训练数据，直接利用 CLIP 的图文匹配能力。

零样本分类流程：
  1. 文本特征构建（只做一次）：
     对每个类别名称，套入模板 "a photo of a {class_name}"
     使用 CLIP 文本编码器编码所有类别，得到 100 个文本特征向量
     （可缓存，整个评估过程只需计算一次）

  2. 图像特征提取：
     对每张测试图片，使用 CLIP 图像编码器提取特征向量

  3. 相似度计算与分类：
     计算图像特征与所有类别文本特征的余弦相似度
     选择相似度最高的类别作为预测结果

算法优势：
  - 无需任何训练数据
  - 无需反向传播
  - 泛化性强，可覆盖训练时未见的类别
  - 是 all few-shot 方法的基准线（baseline）

本类功能：
  - CLIPZeroShot：封装零样本分类的完整流程
  - 提供 predict() 单批预测和 evaluate() 全数据集评估接口
  - 内部处理文本 tokenization、特征归一化、温度缩放等细节
"""

import torch          # PyTorch 张量计算
import clip           # OpenAI CLIP 库：模型加载、tokenize
import numpy as np    # 数值计算
from tqdm import tqdm # 进度条显示


class CLIPZeroShot:
    """
    CLIP 零样本分类器。

    这是所有 few-shot 方法的性能基线（baseline）。
    通过计算图像特征与类别文本特征的余弦相似度进行分类，
    不涉及任何训练或参数更新。

    使用示例：
        model = CLIPZeroShot(backbone='ViT-B/32', device='cuda')
        acc = model.evaluate(test_loader, CIFAR100_CLASSES)
        print(f'Zero-shot accuracy: {acc:.2f}%')
    """

    def __init__(self, backbone='ViT-B/32', device='cuda'):
        """
        初始化 CLIP 零样本模型。

        初始化流程：
          1. 确定计算设备（如果 CUDA 不可用，自动回退到 CPU）
          2. 加载 CLIP 预训练模型和图像预处理函数
             - clip.load() 会自动从 OpenAI 服务器下载权重（首次运行）
             - 权重存储在 ~/.cache/clip/ 目录
          3. 设置为评估模式（eval mode），禁用 dropout/batch norm 等训练行为
          4. 获取模型的数据类型（float16 或 float32），用于后续数据转换

        参数详解：
            backbone : str — CLIP 视觉编码器名称
                'ViT-B/32'   — Vision Transformer Base, patch size 32（默认）
                'ViT-B/16'   — Vision Transformer Base, patch size 16（更高精度）
                'RN50'       — ResNet-50（更快速度）
                'RN101'      — ResNet-101
                'ViT-L/14'   — Vision Transformer Large, patch size 14（最高精度）

            device   : str — 运行设备
                'cuda'  — NVIDIA GPU（推荐）
                'cpu'   — CPU（速度慢）

        属性说明：
            self.model      : CLIP 模型
            self.preprocess : 图像预处理函数（Compose）
            self.dtype      : 模型数据类型（torch.float16 或 torch.float32）
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        # clip.load() 返回模型和预处理函数
        # 模型权重首次加载时会自动下载到 ~/.cache/clip/
        self.model, self.preprocess = clip.load(backbone, device=self.device)
        self.model.eval()  # 切换为评估模式（禁用 Dropout 等训练层）
        # 提取模型的数据类型，用于后续张量类型转换
        self.dtype = next(self.model.visual.parameters()).dtype

    @torch.no_grad()
    def encode_text(self, text):
        """
        将文本列表编码为特征向量。

        处理流程：
          1. 使用 clip.tokenize() 将文本转为 token ID 序列
             - 自动添加 [SOT]（开始标记）和 [EOT]（结束标记）
             - 自动 padding/truncation 到 CLIP 支持的 77 个 token
          2. 送入 CLIP 文本编码器
          3. 返回文本特征向量

        参数说明：
            text : list[str] — 文本列表，如 ["a photo of a dog", "a photo of a cat"]

        返回值：
            torch.Tensor — 文本特征矩阵，形状 [len(text), feature_dim]
                           feature_dim：ViT-B/32 为 512，ViT-L/14 为 768
        """
        tokens = clip.tokenize(text).to(self.device)  # tokenize：文本 → token ID 序列
        return self.model.encode_text(tokens)          # 编码：token ID 序列 → 特征向量

    @torch.no_grad()
    def encode_image(self, images):
        """
        将图像张量编码为特征向量。

        参数说明：
            images : torch.Tensor — 图像张量，形状 [batch_size, 3, 224, 224]

        返回值：
            torch.Tensor — 图像特征矩阵，形状 [batch_size, feature_dim]
        """
        return self.model.encode_image(images)

    def build_class_text_features(self, class_names, prompt_template="a photo of a {}"):
        """
        构建所有类别的文本特征矩阵（归一化后）。

        这是零样本分类的关键预处理步骤：
          1. 对每个类别名称，套入 prompt 模板生成完整的文本描述
          2. 通过 CLIP 文本编码器提取特征
          3. L2 归一化，使特征向量长度为 1（方便后续余弦相似度计算）

        参数详解：
            class_names     : list[str] — 类别名称列表，如 ['dog', 'cat', 'bird']
            prompt_template : str — 文本模板，{} 为类别名称占位符
                模板选择对结果有显著影响，常见模板：
                  "a photo of a {}"          （默认，通用）
                  "a photo of a {}, a type of pet"
                  "a drawing of a {}"
                  "the {}"

        返回值：
            torch.Tensor — 归一化的文本特征矩阵，形状 [num_classes, feature_dim]
                           每行是一个长度为 1 的单位向量
        """
        # 应用模板：["a photo of a dog", "a photo of a cat", ...]
        texts = [prompt_template.format(c) for c in class_names]
        text_features = self.encode_text(texts)

        # L2 归一化：v = v / ||v||，使每个特征向量长度为 1
        # 这样点积就等于余弦相似度
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def predict(self, images, class_names, prompt_template="a photo of a {}"):
        """
        对一批图像进行分类预测。

        完整流程：
          1. 构建类别文本特征（如果之前没有缓存）
          2. 编码输入图像，得到图像特征
          3. L2 归一化图像特征
          4. 计算图像特征与文本特征的矩阵乘法（余弦相似度）
          5. 乘以温度系数 100.0（CLIP 训练时的温度缩放因子）

        参数说明：
            images          : torch.Tensor — 图像批次 [batch_size, 3, 224, 224]
            class_names     : list[str]    — 类别名称列表
            prompt_template : str          — 文本模板

        返回值：
            torch.Tensor — 相似度 logits，形状 [batch_size, num_classes]
                           注意：这不是 softmax 后的概率，而是未归一化的相似度得分
        """
        # 构建归一化的类别文本特征
        text_features = self.build_class_text_features(class_names, prompt_template)

        # 编码并归一化图像特征
        image_features = self.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 计算图像-文本相似度（矩阵乘法 = 余弦相似度 × 温度缩放）
        # 温度系数 100.0 来源于 CLIP 训练时的 logit_scale
        logits = (image_features @ text_features.T) * 100.0
        return logits

    @torch.no_grad()
    def evaluate(self, test_loader, class_names, prompt_template="a photo of a {}"):
        """
        在完整测试集上评估零样本分类准确率。

        评估流程：
          1. 一次性构建所有类别的文本特征（整个测试过程共享）
          2. 遍历测试集的每个 batch：
             a. 将图像移至 GPU
             b. 提取并归一化图像特征
             c. 计算与文本特征的余弦相似度
             d. 取相似度最高的类别作为预测结果
          3. 汇总所有预测结果与真实标签
          4. 计算准确率 = 预测正确的样本数 / 总样本数 × 100

        参数说明：
            test_loader     : DataLoader — 测试集数据加载器
                               每个 batch 返回 (images, labels)
            class_names     : list[str]  — 100 个类别名称
            prompt_template : str — 文本模板

        返回值：
            float — 分类准确率（百分比），范围 [0, 100]

        性能说明：
            对于 CIFAR-100 测试集（10000 张）：
              - ViT-B/32 约 5 秒完成评估
              - ViT-B/16 约 8 秒完成评估
        """
        # 一次性构建类别文本特征（只需要计算一次）
        text_features = self.build_class_text_features(class_names, prompt_template)

        all_preds = []   # 存储所有预测标签
        all_labels = []  # 存储所有真实标签

        # 遍历测试集的每个 batch
        for images, labels in tqdm(test_loader, desc='Evaluating CLIP Zero-shot'):
            images = images.to(self.device)  # 将图像数据移至 GPU

            # 提取图像特征
            image_features = self.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # 计算相似度并取最大值对应的类别索引
            logits = image_features @ text_features.T  # 矩阵乘法
            preds = logits.argmax(dim=-1)               # 取相似度最高的类别

            all_preds.append(preds.cpu())    # 将结果移回 CPU
            all_labels.append(labels)        # 真实标签

        # 拼接所有 batch 的结果
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # 计算准确率：预测正确的比例 × 100
        acc = (all_preds == all_labels).float().mean().item() * 100.0
        return acc
