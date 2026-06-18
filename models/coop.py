"""
=====================================================================
  CoOp (Context Optimization) —— 可学习 Prompt 向量模型
=====================================================================

论文来源：
  Learning to Prompt for Vision-Language Models
  (Zhou et al., IJCV 2022)

核心思想：
  CoOp 将 CLIP 中人工设计的 prompt 模板（如 "a photo of a {class}"）
  替换为 **可学习的连续向量**（learnable continuous vectors），
  通过 few-shot 标注样本优化这些向量，使 prompt 更好地适配下游任务。

为什么 CoOp 有效？
  人工设计的 prompt 如 "a photo of a" 是针对通用场景设计的，
  对于特定数据集不一定是最优的。例如：
  - CIFAR-100（小图 32×32）：可能需要 "a small photo of" 强调尺寸
  - 医学图像：可能需要 "a CT scan of" 等专业描述
  CoOp 通过梯度下降自动发现最适合当前数据集的 prompt 表达方式。

原始 CLIP prompt 结构：
  [SOT]  a_photo_of_a  {class_name}  [EOT]
  [SOT]  [token]       [token]       [EOT]   ← 固定的 token 嵌入

CoOp prompt 结构：
  [SOT]  [V1] [V2] ... [V16]  {class_name}  [EOT]
  [SOT]  [学习向量]              [token]       [EOT]   ← V1~V16 可学习

训练策略：
  - 冻结 CLIP 文本编码器和图像编码器的全部参数
  - 仅优化 prompt 向量（context vectors），数量由 n_ctx 控制
  - 可学习参数量：n_ctx × embedding_dim
    以 ViT-B/32 为例：16 × 512 = 8,192 个参数
  - 使用 SGD with Momentum 优化器（原论文推荐）
  - 余弦退火学习率调度（CosineAnnealingLR）

本模块包含三个主要组件：
  1. CoOp 类      — CoOp 模型定义（前向传播逻辑）
  2. train_coop   — CoOp 训练函数
  3. evaluate_coop — CoOp 评估函数
"""

import torch
import torch.nn as nn  # PyTorch 神经网络模块：Linear, Parameter, functional
import clip            # OpenAI CLIP 库
from tqdm import tqdm  # 进度条


class CoOp(nn.Module):
    """
    CoOp 模型。

    学习一组连续 prompt 向量（context vectors），替换人工设计的 prompt。
    这些向量插入到文本编码器的输入 token 序列中，在类别名称之前。

    模型结构：
      输入：[SOT] [V1] [V2] ... [Vn_ctx] [class_tokens] [EOT]
              ↑      ↑                      ↑
          固定嵌入  可学习参数(梯度更新)   固定嵌入

    参数规模：
      以 n_ctx=16, ViT-B/32 (ctx_dim=512) 为例：
      可学习参数 = 16 × 512 = 8,192（约 8K 参数）

    使用示例：
        model = CoOp(backbone='ViT-B/32', n_ctx=16, csc=False, device='cuda')
        logits = model(images, class_names)
    """

    def __init__(self, backbone='ViT-B/32', n_ctx=16, csc=False, device='cuda'):
        """
        初始化 CoOp 模型。

        初始化流程：
          1. 确定设备（自动回退到 CPU）
          2. 加载 CLIP 模型（关闭 JIT 以便操作内部模块）
          3. 冻结 CLIP 全部参数（只训练 prompt 向量）
          4. 初始化可学习的 prompt 向量

        参数详解：
            backbone : str — CLIP 模型名称
                'ViT-B/32'   — Vision Transformer Base, patch 32（默认）
                'ViT-B/16'   — 更高精度
                'RN50'       — ResNet-50

            n_ctx    : int  — 可学习 prompt 向量的数量（上下文长度）
                论文推荐：16
                n_ctx 越大 → 参数越多 → 表达能力越强 → 更容易过拟合
                n_ctx 越小 → 参数越少 → 欠拟合风险增加

            csc      : bool — Class-Specific Context
                False — 所有类别共享同一组 prompt（推荐，防止过拟合）
                True  — 每个类别学习独立的 prompt（需要更多数据）

            device   : str  — 运行设备

        重要属性：
            self.ctx : nn.Parameter — 可学习的 prompt 向量
                形状取决于 csc：
                  csc=False → [n_ctx, ctx_dim]
                  csc=True  → [100, n_ctx, ctx_dim]（每个类别独立）
        """
        super().__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.n_ctx = n_ctx    # 可学习 prompt 数量
        self.csc = csc         # 是否使用类别专属 prompt

        # 加载 CLIP 模型
        # jit=False 是关键：关闭 JIT 编译，才能访问内部模块
        # （如 token_embedding, transformer, ln_final, text_projection 等）
        self.clip_model, self.preprocess = clip.load(backbone, device=self.device, jit=False)
        # 转为 float32 进行训练（CLIP 默认可能使用 float16）
        self.clip_model = self.clip_model.float()

        # 冻结 CLIP 的所有参数，使其不参与梯度更新
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 获取文本编码器的输出特征维度
        # ln_final 是文本编码器最后一层 LayerNorm 的权重
        ctx_dim = self.clip_model.ln_final.weight.shape[0]  # ViT-B/32 → 512
        self.dtype = self.clip_model.dtype

        # ===== 初始化可学习的 prompt 向量 =====
        # 使用 nn.Parameter 包装，使其被 PyTorch 自动追踪梯度
        if csc:
            # Class-Specific Context：每个类别学习 100 组不同的 prompt
            # 形状：[100, n_ctx, ctx_dim]
            self.ctx = nn.Parameter(torch.empty(100, n_ctx, ctx_dim))
        else:
            # 共享 Context：所有类别使用同一组 prompt
            # 形状：[n_ctx, ctx_dim]
            self.ctx = nn.Parameter(torch.empty(n_ctx, ctx_dim))

        # 使用正态分布初始化（均值 0，标准差 0.02）
        # 这是神经网络参数初始化的常用策略
        nn.init.normal_(self.ctx, std=0.02)

    def forward(self, images, class_names):
        """
        CoOp 的前向传播函数。

        完整流程：
          1. 图像编码：使用 CLIP 图像编码器提取图像特征
          2. Prompt 构建：为每个类别构建完整的 prompt 嵌入序列
             a. SOT 嵌入（开始标记）
             b. 可学习的 context 向量（核心）
             c. 类别名称的 token 嵌入
             d. EOT 嵌入（结束标记）
             e. Padding 补齐到 77 个 token
          3. 文本编码：将 prompt 序列送入 CLIP 文本编码器
          4. 相似度计算：图像特征 × 文本特征 得到分类 logits

        参数说明：
            images      : torch.Tensor — 图像批次，形状 [B, 3, 224, 224]
            class_names : list[str]    — 类别名称列表，长度 C（CIFAR-100 为 100）

        返回值：
            torch.Tensor — 分类 logits，形状 [B, C]
                           B = batch size, C = num classes (100)
        """
        # =============================================================
        # 1. 图像编码分支
        # =============================================================
        # 使用 CLIP 图像编码器提取特征
        image_features = self.clip_model.encode_image(images)
        # L2 归一化，使特征向量长度为 1
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        num_classes = len(class_names)  # CIFAR-100 → 100

        # =============================================================
        # 2. Prompt 序列构建
        # =============================================================
        # 目标：为每个类别构造一个完整的 prompt 序列
        # 序列结构（从左到右）：
        #   [SOT_embed(1)] [ctx_embeds(n_ctx)] [class_embeds(len_c)] [EOT_embed(1)]
        #   最后补齐到 77 个 token

        # 2a. 对类别名称进行 tokenize（不加 "a photo of a" 前缀）
        # clip.tokenize 返回形状 [num_classes, 77] 的 token ID 矩阵
        token_ids = clip.tokenize(class_names).to(self.device)

        # 找到每个序列中 EOT token (ID=49407) 的位置
        # EOT 是 CLIP tokenizer 中的结束标记
        orig_eot_positions = (token_ids == 49407).float().argmax(dim=-1)

        # 2b. 获取 SOT（开始标记）的嵌入向量
        # SOT token ID = 49406
        sos_embeds = self.clip_model.token_embedding(
            torch.full((num_classes, 1), 49406, dtype=torch.long, device=self.device)
        )  # 形状: [100, 1, 512]

        # 2c. 获取可学习的 context 向量
        # 如果是共享模式(csc=False)，将 [n_ctx, dim] 扩展到 [100, n_ctx, dim]
        ctx_embeds = self.ctx.unsqueeze(0).expand(num_classes, -1, -1).to(self.device)
        # 形状: [100, n_ctx, 512]

        # 2d. 提取类别名称部分的 token 嵌入
        # 去除 token_ids 中的 SOT(token 0) 和 EOT(token 最后)，只保留类别名称部分
        # 注意：不同类别名称的 token 长度可能不同（如 "apple" vs "aquarium_fish"）
        class_part_list = []
        for i in range(num_classes):
            eot_pos = int(orig_eot_positions[i])
            # 取 [第 1 个 token : EOT 位置) 之间的部分（跳过 SOT）
            class_token_ids = token_ids[i, 1:eot_pos]
            class_part_list.append(class_token_ids.unsqueeze(0))

        # 将不等长的类别名称补齐到相同长度
        # 找出最长的类别名称的 token 长度
        max_class_len = max(t.shape[1] for t in class_part_list)
        padded_class_ids = []
        for t in class_part_list:
            if t.shape[1] < max_class_len:
                # 用 0 填充到 max_class_len（0 是 padding token ID）
                pad = torch.zeros(1, max_class_len - t.shape[1], dtype=torch.long, device=self.device)
                t = torch.cat([t, pad], dim=1)
            padded_class_ids.append(t)
        class_ids = torch.cat(padded_class_ids, dim=0)  # 形状: [100, max_class_len]

        # 嵌入类别名称 token
        # 注意：对于 padding 位置（token ID=0），我们显式传入 0 保持一致的嵌入
        zero_ids = torch.zeros_like(class_ids)
        class_embeds = torch.where(
            (class_ids != 0).unsqueeze(-1),  # 非 padding 位置
            self.clip_model.token_embedding(class_ids),  # 正常嵌入
            self.clip_model.token_embedding(zero_ids)    # padding 位置嵌入 0
        )
        # 形状: [100, max_class_len, 512]

        # 2e. 获取 EOT（结束标记）的嵌入向量
        # EOT token ID = 49407
        eos_embeds = self.clip_model.token_embedding(
            torch.full((num_classes, 1), 49407, dtype=torch.long, device=self.device)
        )
        # 形状: [100, 1, 512]

        # 2f. 拼接完整的 prompt 序列
        # 顺序: SOT → Context → Class Name → EOT
        prompt = torch.cat([sos_embeds, ctx_embeds, class_embeds, eos_embeds], dim=1)
        # 形状: [100, 1 + n_ctx + max_class_len + 1, 512]
        n_cur = prompt.shape[1]  # 当前序列长度

        # 2g. 补齐到 CLIP 要求的最大序列长度（77）
        # CLIP 的文本编码器固定输入长度为 77
        max_len = 77
        if n_cur < max_len:
            pad_len = max_len - n_cur
            # 用 token ID=0 的嵌入填充剩余位置
            pad_token_ids = torch.zeros(num_classes, pad_len, dtype=torch.long, device=self.device)
            pad_embeds = self.clip_model.token_embedding(pad_token_ids)
            prompt = torch.cat([prompt, pad_embeds], dim=1)
        # 最终形状: [100, 77, 512]

        # =============================================================
        # 3. 文本编码（使用 CLIP 文本编码器）
        # =============================================================
        # 添加位置编码（positional embedding），与 CLIP 原始前向一致
        prompt = prompt + self.clip_model.positional_embedding[:prompt.shape[1]].unsqueeze(0)

        # 转换为文本编码器需要的格式：[seq_len, batch, dim]
        # 注意：CLIP 的 transformer 使用这个特定的维度顺序
        prompt = prompt.permute(1, 0, 2)  # [77, 100, 512]

        # 通过 CLIP 的 transformer 编码器
        x = self.clip_model.transformer(prompt)
        x = x.permute(1, 0, 2)            # 转回 [100, 77, 512]
        x = self.clip_model.ln_final(x)   # 最后一层 LayerNorm

        # 取出 EOT 位置的特征作为整个序列的文本表示
        # CLIP 使用 EOT 位置的特征作为文本特征（这是 CLIP 的设计选择）
        eot_pos = 1 + self.n_ctx + max_class_len  # EOT 在序列中的位置
        text_features = x[torch.arange(num_classes), eot_pos]  # 形状: [100, 512]

        # 应用 text_projection（CLIP 文本编码器的最后线性投影层）
        text_features = text_features @ self.clip_model.text_projection
        # L2 归一化
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # =============================================================
        # 4. 计算图像-文本相似度
        # =============================================================
        # 矩阵乘法得到分类 logits
        logits = image_features @ text_features.T * 100.0  # 温度系数 100.0
        # 形状: [batch_size, 100]
        return logits

    @torch.no_grad()
    def predict(self, images, class_names):
        """推理时调用的预测接口，与 forward 相同但标记为无梯度计算"""
        return self.forward(images, class_names)


def train_coop(model, train_loader, val_loader, class_names, config):
    """
    训练 CoOp 模型的 prompt 向量。

    训练设置（遵循原论文）：
      - 优化器：SGD with Momentum（momentum=0.9）
      - 学习率：config.coop_lr（默认 0.002）
      - 权重衰减：config.coop_weight_decay（默认 1e-4）
      - 学习率调度：余弦退火（CosineAnnealingLR）
      - 损失函数：交叉熵（CrossEntropyLoss）

    训练流程：
      每个 epoch：
        1. 遍历训练集的每个 batch
        2. 前向传播计算 logits
        3. 计算交叉熵损失
        4. 反向传播计算梯度
        5. 优化器更新 prompt 向量
      训练结束后在测试集上评估最终准确率

    参数说明：
        model       : CoOp    — CoOp 模型实例
        train_loader: DataLoader — 训练数据加载器
        val_loader  : DataLoader — 测试数据加载器
        class_names : list[str]  — 100 个类别名称
        config      : Config    — 配置对象（包含训练超参数）

    返回值：
        best_acc  : float — 测试集上的最佳准确率（%）
        ctx_state : torch.Tensor — 训练好的 prompt 向量（在 CPU 上），
                    用于保存 checkpoint
    """
    # 设置优化器：只优化 model.ctx（可学习 prompt 向量）
    # 论文推荐使用 SGD with Momentum
    optimizer = torch.optim.SGD(
        [model.ctx],               # 只优化 prompt 向量
        lr=config.coop_lr,
        momentum=0.9,              # 动量系数
        weight_decay=config.coop_weight_decay
    )

    # 余弦退火学习率调度器
    # T_max = coop_epochs：学习率在 0 → T_max 之间从初始值下降到 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.coop_epochs)

    # 分类任务使用交叉熵损失
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    # 保存 prompt 向量的副本，用于返回和 checkpoint 存储
    best_ctx = model.ctx.detach().cpu().clone()

    # Epoch 循环
    for epoch in range(config.coop_epochs):
        model.train()  # 设置为训练模式

        # Batch 循环
        for images, labels in tqdm(train_loader, desc=f'CoOp Epoch {epoch+1}/{config.coop_epochs}', leave=False):
            images = images.to(model.device)
            labels = labels.to(model.device)

            # 前向传播
            logits = model(images, class_names)
            # 计算损失
            loss = criterion(logits, labels)

            # 反向传播和优化
            optimizer.zero_grad()  # 清空上一步的梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数

        # 更新学习率
        scheduler.step()

    # 训练结束后在测试集上评估
    with torch.no_grad():
        best_acc = evaluate_coop(model, val_loader, class_names)

    # 返回最终的 prompt 向量（用于保存 checkpoint）
    return best_acc, model.ctx.detach().cpu()


@torch.no_grad()
def evaluate_coop(model, test_loader, class_names):
    """
    在测试集上评估 CoOp 模型的分类准确率。

    评估流程与 Zero-shot 类似，但使用 CoOp 学到的 prompt 向量
    替代人工设计的 prompt 模板来计算文本特征。

    参数说明：
        model       : CoOp       — CoOp 模型
        test_loader : DataLoader — 测试数据加载器
        class_names : list[str]  — 100 个类别名称

    返回值：
        float — 分类准确率（百分比），范围 [0, 100]
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []

    # 遍历测试集
    for images, labels in tqdm(test_loader, desc='Evaluating CoOp'):
        images = images.to(model.device)
        logits = model(images, class_names)
        preds = logits.argmax(dim=-1)  # 取最大 logit 对应的类别

        all_preds.append(preds.cpu())
        all_labels.append(labels)

    # 拼接并计算准确率
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    acc = (all_preds == all_labels).float().mean().item() * 100.0
    return acc
