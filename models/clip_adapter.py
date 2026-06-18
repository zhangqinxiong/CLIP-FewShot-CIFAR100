"""
=====================================================================
  CLIP-Adapter —— 特征适配器微调模型
=====================================================================

论文来源：
  CLIP-Adapter: Better Vision-Language Models with Feature Adapters
  (Gao et al., arXiv 2021)

核心思想：
  在 CLIP 的图像编码器和文本编码器后分别添加轻量级的 Adapter 模块，
  通过残差连接微调特征表示，以适应下游任务的分布偏移。

  与 CoOp 的区别：
    - CoOp 修改的是输入（prompt 模板），在文本嵌入空间操作
    - CLIP-Adapter 修改的是特征（feature），在特征表示空间操作
    - 两者可以互补（有工作结合两者取得了更好的效果）

Adapter 结构（瓶颈设计，bottleneck architecture）：
    Linear(feat_dim, reduce_dim)  →  降维（减少参数量）
        ↓
    ReLU()                         →  非线性激活
        ↓
    Linear(reduce_dim, feat_dim)  →  升维回原始维度

残差融合（关键设计）：
    output = alpha * adapter(x) + (1 - alpha) * x

  残差连接的作用：
    1. 缓解梯度消失：梯度可以直接通过恒等路径传播
    2. 保留预训练知识：大部分信息直接通过恒等路径保留
    3. 防止过拟合：Adapter 的瓶颈结构限制了参数量（强正则化）

训练策略：
  - 冻结 CLIP 的全部原始参数（约 1.5 亿参数）
  - 仅训练两个 Adapter 模块：
    - 图像端 Adapter（图像特征维度 → 256 → 图像特征维度）
    - 文本端 Adapter（文本特征维度 → 256 → 文本特征维度）
  - 总可学习参数约 524K（对于 ViT-B/32）
  - 使用 Adam 优化器，余弦退火学习率

本模块包含三个主要组件：
  1. Adapter 类    — 轻量瓶颈适配器模块
  2. CLIPAdapter 类 — CLIP-Adapter 模型（包含图像和文本 Adapter）
  3. train_adapter  — 训练函数
  4. evaluate_adapter — 评估函数
"""

import torch
import torch.nn as nn  # 神经网络模块
import clip            # OpenAI CLIP
from tqdm import tqdm  # 进度条


class Adapter(nn.Module):
    """
    Adapter 瓶颈适配器模块。

    这是一个轻量级特征适配器，采用瓶颈结构（bottleneck architecture），
    先降维再升维，中间使用 ReLU 激活。

    数学定义：
        f(x) = alpha * W_up(ReLU(W_down(x))) + (1 - alpha) * x

        其中：
          - W_down : ℝ^in_dim → ℝ^reduce_dim（降维线性层）
          - W_up   : ℝ^reduce_dim → ℝ^in_dim（升维线性层）
          - alpha  : 残差融合系数

    参数量计算（ViT-B/32, in_dim=512, reduce_dim=256）：
        W_down : 512 × 256 = 131,072
        W_up   : 256 × 512 = 131,072
        合计   : 262,144（约 262K 参数）
    """

    def __init__(self, in_dim=512, reduce_dim=256, alpha=0.2):
        """
        初始化 Adapter 模块。

        参数详解：
            in_dim     : int   — 输入特征维度
                ViT-B/32 → 512
                ViT-B/16 → 512
                RN50     → 1024
            reduce_dim : int   — 瓶颈层维度，一般为 in_dim 的 1/2 到 1/4
                默认 256，增大可提升拟合能力但增加过拟合风险
            alpha      : float — 残差融合系数
                范围 [0, 1]
                alpha=0 → 恒等映射，退化为原始 CLIP 特征
                alpha=1 → 完全使用 Adapter 输出
                默认 0.2 保留 80% 原始特征 + 20% Adapter 特征

        网络结构：
            Sequential(
              (0): Linear(in_dim → reduce_dim)    # 降维
              (1): ReLU(inplace=True)             # 激活
              (2): Linear(reduce_dim → in_dim)    # 升维
            )
        """
        super().__init__()
        self.alpha = alpha  # 残差融合系数

        # 瓶颈网络：降维 → 激活 → 升维
        self.net = nn.Sequential(
            nn.Linear(in_dim, reduce_dim),        # 降维层
            nn.ReLU(inplace=True),                # 非线性激活（inplace=True 节省显存）
            nn.Linear(reduce_dim, in_dim),        # 升维层
        )

        # 使用 Xavier 均匀初始化 + 零偏置
        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        """
        使用 Xavier 均匀初始化初始化线性层权重，偏置设为 0。

        Xavier 初始化（Glorot 初始化）：
          权重从 Uniform(-bound, bound) 采样
          bound = sqrt(6 / (fan_in + fan_out))
          保持前向和反向传播中梯度的方差稳定
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)  # Xavier 均匀初始化
            nn.init.zeros_(m.bias)             # 偏置初始化为 0

    def forward(self, x):
        """
        Adapter 前向传播。

        计算公式：
            output = alpha * adapter(x) + (1 - alpha) * x

        参数说明：
            x : torch.Tensor — 输入特征，形状 [batch_size, in_dim]

        返回值：
            torch.Tensor — 适配后的特征，形状与输入相同 [batch_size, in_dim]
        """
        # 瓶颈适配：降维 → ReLU → 升维
        adapted = self.net(x)
        # 残差融合：alpha * 适配特征 + (1-alpha) * 原始特征
        return self.alpha * adapted + (1 - self.alpha) * x


class CLIPAdapter(nn.Module):
    """
    CLIP-Adapter 模型。

    在 CLIP 的图像编码器和文本编码器后各添加一个 Adapter，
    通过残差连接微调特征表示。

    训练时，CLIP 的原始参数完全冻结，只更新两个 Adapter 的权重。

    使用示例：
        model = CLIPAdapter(backbone='ViT-B/32', reduce_dim=256, alpha=0.2, device='cuda')
        logits = model(images, class_names)
    """

    def __init__(self, backbone='ViT-B/32', reduce_dim=256, alpha=0.2, device='cuda'):
        """
        初始化 CLIP-Adapter 模型。

        初始化流程：
          1. 确定设备
          2. 加载 CLIP 模型并转为 float32
          3. 冻结 CLIP 全部参数
          4. 动态获取图像和文本的特征维度
          5. 创建图像端和文本端的 Adapter

        参数详解：
            backbone   : str   — CLIP 模型名称
            reduce_dim : int   — Adapter 瓶颈层维度
            alpha      : float — 残差融合系数
            device     : str   — 运行设备
        """
        super().__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha  # 残差融合系数

        # ===== 加载 CLIP 模型 =====
        self.clip_model, self.preprocess = clip.load(backbone, device=self.device)
        # 转为 float32 训练（CLIP 默认可能使用 float16）
        self.clip_model = self.clip_model.float()

        # ===== 冻结 CLIP 全部参数 =====
        # 训练时只有 Adapter 的参数会更新
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # ===== 动态获取特征维度 =====
        # 不同 backbone 的特征维度不同，因此需要动态获取
        with torch.no_grad():
            # 图像特征维度：用一张虚拟图片过一次编码器
            dummy = torch.randn(1, 3, 224, 224, device=self.device)
            self.img_dim = self.clip_model.encode_image(dummy).shape[-1]
            # 文本特征维度：用一个虚拟文本过一次编码器
            txt_tokens = clip.tokenize(['a']).to(self.device)
            self.txt_dim = self.clip_model.encode_text(txt_tokens).shape[-1]

        # ===== 创建图像端和文本端的 Adapter =====
        # reduce_dim 取用户指定值和特征维度一半中的较大值
        # 保证瓶颈层不会太小，保留足够的表达能力
        img_reduce = max(self.img_dim // 2, reduce_dim)
        txt_reduce = max(self.txt_dim // 2, reduce_dim)

        self.img_adapter = Adapter(self.img_dim, img_reduce, alpha).to(self.device)
        self.txt_adapter = Adapter(self.txt_dim, txt_reduce, alpha).to(self.device)

    def forward(self, images, class_names):
        """
        CLIP-Adapter 前向传播。

        完整流程：
          1. 图像分支：
             a. CLIP 图像编码器提取原始图像特征
             b. L2 归一化
             c. 通过图像 Adapter 进行特征适配（残差连接）
             d. 再次 L2 归一化
          2. 文本分支：
             a. 使用模板 "a photo of a {class}" 格式化类别名称
             b. CLIP 文本编码器提取原始文本特征
             c. L2 归一化
             d. 通过文本 Adapter 进行特征适配（残差连接）
             e. 再次 L2 归一化
          3. 计算图像-文本相似度（矩阵乘法 × 温度系数）

        参数说明：
            images      : torch.Tensor — 图像批次 [B, 3, 224, 224]
            class_names : list[str]    — 类别名称列表，长度 C

        返回值：
            torch.Tensor — 分类 logits，形状 [B, C]
        """
        # =============================================================
        # 图像分支
        # =============================================================
        # 1. CLIP 图像编码器提取原始特征
        image_features = self.clip_model.encode_image(images).float()
        # 2. L2 归一化
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 3. 图像 Adapter 适配（含残差连接）
        image_features = self.img_adapter(image_features)
        # 4. 再次 L2 归一化（适配后的特征可能改变长度）
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # =============================================================
        # 文本分支
        # =============================================================
        # 1. 构建文本描述（使用标准模板）
        texts = [f"a photo of a {c}" for c in class_names]
        # 2. tokenize 并移至 GPU
        tokens = clip.tokenize(texts).to(self.device)
        # 3. CLIP 文本编码器提取原始特征
        text_features = self.clip_model.encode_text(tokens).float()
        # 4. L2 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # 5. 文本 Adapter 适配（含残差连接）
        text_features = self.txt_adapter(text_features)
        # 6. 再次 L2 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # =============================================================
        # 分类：计算图像-文本相似度
        # =============================================================
        # 图像特征 @ 文本特征^T = 余弦相似度矩阵
        logits = image_features @ text_features.T * 100.0  # 温度系数
        return logits

    @torch.no_grad()
    def predict(self, images, class_names):
        """推理接口，与 forward 相同但标记为无梯度计算"""
        return self.forward(images, class_names)


def train_adapter(model, train_loader, val_loader, class_names, config):
    """
    训练 CLIP-Adapter 模型的图像和文本 Adapter 模块。

    训练设置：
      - 优化器：Adam（自适应学习率，适合 Adapter 微调）
      - 学习率：config.adapter_lr（默认 0.001）
      - 权重衰减：config.adapter_weight_decay（默认 1e-4）
      - 学习率调度：余弦退火（CosineAnnealingLR）
      - 损失函数：交叉熵（CrossEntropyLoss）

    训练流程：
      每个 epoch：
        1. 遍历训练集 batch
        2. 前向传播计算 logits
        3. 交叉熵损失
        4. 反向传播（只更新 Adapter 权重）
        5. 优化器步进
      训练结束后在测试集上评估准确率

    参数说明：
        model       : CLIPAdapter — CLIP-Adapter 模型
        train_loader: DataLoader  — 训练数据加载器
        val_loader  : DataLoader  — 测试数据加载器
        class_names : list[str]   — 类别名称列表
        config      : Config     — 配置对象

    返回值：
        best_acc : float — 测试集准确率（%）
        state    : dict  — Adapter 权重字典（用于保存 checkpoint）
    """
    # 收集图像 Adapter 和文本 Adapter 的参数
    # 注意：CLIP 的原始参数 requires_grad=False，不会出现在这里
    params = list(model.img_adapter.parameters()) + list(model.txt_adapter.parameters())

    # Adam 优化器：适合 Adapter 微调（自适应学习率）
    optimizer = torch.optim.Adam(
        params,
        lr=config.adapter_lr,
        weight_decay=config.adapter_weight_decay
    )

    # 余弦退火学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.adapter_epochs)

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # Epoch 循环
    for epoch in range(config.adapter_epochs):
        model.train()  # 切换为训练模式

        # Batch 循环
        for images, labels in tqdm(train_loader, desc=f'CLIP-Adapter Epoch {epoch+1}/{config.adapter_epochs}', leave=False):
            images = images.to(model.device)
            labels = labels.to(model.device)

            logits = model(images, class_names)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 更新学习率
        scheduler.step()

    # 训练结束后在测试集上评估
    best_acc = evaluate_adapter(model, val_loader, class_names)

    # 提取 Adapter 的权重字典（用于保存 checkpoint）
    # 只保存 Adapter 相关权重，不保存 CLIP 原始权重
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if 'img_adapter' in k or 'txt_adapter' in k}
    return best_acc, state


@torch.no_grad()
def evaluate_adapter(model, test_loader, class_names):
    """
    在测试集上评估 CLIP-Adapter 的分类准确率。

    评估流程：
      1. 遍历测试集的每个 batch
      2. 模型前向传播得到分类 logits
      3. 取 argmax 得到预测类别
      4. 计算准确率

    参数说明：
        model       : CLIPAdapter — CLIP-Adapter 模型
        test_loader : DataLoader  — 测试数据加载器
        class_names : list[str]   — 100 个类别名称

    返回值：
        float — 分类准确率（%）
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    for images, labels in tqdm(test_loader, desc='Evaluating CLIP-Adapter'):
        images = images.to(model.device)
        logits = model(images, class_names)
        preds = logits.argmax(dim=-1)

        all_preds.append(preds.cpu())
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item() * 100.0
    return acc
