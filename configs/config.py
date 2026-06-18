"""
=====================================================================
  全局配置文件 —— 集中管理所有实验的超参数和路径设置
=====================================================================

功能说明：
  本文件定义了 Config 类，作为整个项目的唯一配置中心。
  所有超参数（数据集设置、模型参数、训练策略、输出路径等）
  集中在此处管理，避免超参数散落在各个脚本中。

设计原则：
  - 单一职责：所有配置集中在本文件，修改参数只需编辑此处
  - 默认最优：参数默认值参照原论文推荐设置
  - 易于扩展：添加新参数只需在 Config 类中增加属性

使用方式：
  from configs.config import Config
  config = Config()
  print(config.backbone)         # 读取配置
  config.coop_epochs = 100       # 修改配置（运行时覆盖）

配置分类：
  1. 数据集设置      dataset, data_root
  2. Few-shot 设置   shots
  3. CLIP 主干网络    backbone, device
  4. CoOp 超参数      coop_n_ctx, coop_epochs, coop_lr ...
  5. CLIP-Adapter 超参数 adapter_reduce_dim, adapter_alpha ...
  6. 评估设置         test_batch_size
  7. 输出设置         output_dir, results_file
"""


class Config:
    """
    全局配置类。

    所有属性均为类属性（class attributes），
    实例化后可直接通过实例访问。

    使用示例：
        config = Config()
        print(config.dataset)         # 输出: 'cifar100'
        print(config.shots)           # 输出: [1, 2, 4, 8, 16]
        print(config.coop_lr)         # 输出: 0.002
    """

    # =============================================================
    # 数据集设置
    # =============================================================
    # dataset  : str  — 数据集名称。当前仅支持 'cifar100'，
    #                   预留字段便于未来扩展其他数据集（如 ImageNet、Flowers102 等）
    dataset = 'cifar100'

    # data_root : str — 数据集在本地磁盘的存储路径
    #   首次运行时，torchvision 会自动下载 CIFAR-100 到此目录
    #   下载后的目录结构：
    #     datasets/cifar100/
    #       ├── cifar-100-python/       # 原始数据文件
    #       └── cifar-100-python.tar.gz # 压缩包
    data_root = './datasets/cifar100'

    # =============================================================
    # Few-shot 设置
    # =============================================================
    # shots : list[int] — 每个类别使用的训练样本数量列表
    #   含义：例如 shots=[1, 2, 4, 8, 16] 表示分别使用
    #         每类 1、2、4、8、16 张训练图片进行实验
    #   总训练样本数 = shot × 类别数（CIFAR-100 为 100）
    #   例如 16-shot → 1600 张训练图片
    #
    # 论文标准配置：[1, 2, 4, 8, 16]
    # 如需自定义：  shots = [1, 4, 16]  或  shots = [1, 2, 4, 8, 16, 32]
    shots = [1, 2, 4, 8, 16]

    # =============================================================
    # CLIP 主干网络
    # =============================================================
    # backbone : str — CLIP 视觉编码器模型名称
    #   可选值（按性能/速度排序）：
    #     'RN50'      — ResNet-50，最快但精度最低
    #     'RN101'     — ResNet-101
    #     'RN50x4'    — ResNet-50 × 4（效率通道缩放）
    #     'RN50x16'   — ResNet-50 × 16（更大规模）
    #     'RN50x64'   — ResNet-50 × 64（最大规模，需要高显存）
    #     'ViT-B/32'  — ViT Base, patch size 32（默认推荐）
    #     'ViT-B/16'  — ViT Base, patch size 16（精度更高）
    #     'ViT-L/14'  — ViT Large, patch size 14（精度最高，显存需求大）
    #
    # 显存需求参考（batch_size=32）：
    #   ViT-B/32  ~ 2GB
    #   ViT-B/16  ~ 4GB
    #   ViT-L/14  ~ 12GB
    backbone = 'ViT-B/32'

    # device : str — 计算设备
    #   'cuda' — 使用 GPU（需安装 CUDA 版 PyTorch）
    #   'cpu'  — 使用 CPU（速度慢，仅用于测试）
    device = 'cuda'

    # =============================================================
    # CoOp (Context Optimization) 超参数
    # =============================================================
    # coop_n_ctx : int — 可学习 prompt 向量的数量（上下文长度）
    #   含义：在类别名称前插入 M 个可学习的连续向量
    #   论文推荐值：16（经过实验验证的最优值）
    #   增大 n_ctx → 更多可学习参数 → 可能过拟合（尤其 shot 少时）
    #   减小 n_ctx → 参数更少 → 表达能力受限
    #
    # 序列结构：
    #   [SOT_embed, V1, V2, ..., V16, class_token_embeds, EOT_embed]
    #   其中 V1~V16 是可学习的 prompt 向量
    coop_n_ctx = 16

    # coop_epochs : int — CoOp 训练的最大轮数
    #   论文中 50 轮足够收敛，shot 较少时可以适当减少
    coop_epochs = 50

    # coop_batch_size : int — CoOp 训练的批大小
    #   受 GPU 显存限制，ViT-B/32 推荐 32，ViT-B/16 推荐 16
    coop_batch_size = 32

    # coop_lr : float — CoOp 的学习率
    #   论文推荐使用 SGD 优化器，学习率 0.002
    #   如果使用 Adam 优化器，建议降低学习率至 0.0001~0.0005
    coop_lr = 0.002

    # coop_weight_decay : float — 权重衰减系数（L2 正则化）
    #   防止过拟合，尤其 shot 较少时效果明显
    coop_weight_decay = 1e-4

    # coop_csc : bool — 是否使用类别专属 prompt（Class-Specific Context）
    #   False：所有类别共享同一组 prompt 向量（默认，参数量少）
    #   True ：每个类别学习独立的 prompt 向量（参数量 × 100）
    #   CSC 模式适用场景：每类样本较多时（如 8-shot 以上）
    coop_csc = False

    # =============================================================
    # CLIP-Adapter 超参数
    # =============================================================
    # adapter_reduce_dim : int — Adapter 瓶颈层维度
    #   Adapter 结构：Linear(in_dim → reduce_dim) → ReLU → Linear(reduce_dim → in_dim)
    #   默认值为 256，特征维度（ViT-B/32 为 512）的一半
    #   增大 reduce_dim → 更多可学习参数 → 拟合能力增强
    #   减小 reduce_dim → 更少参数 → 正则化效果更强
    adapter_reduce_dim = 256

    # adapter_alpha : float — 残差融合系数
    #   output = alpha * adapter(x) + (1 - alpha) * x
    #   alpha 控制 Adapter 输出与原始特征的混合比例
    #   推荐范围：[0.1, 0.5]
    #   alpha=0.0 → 完全使用原始 CLIP 特征（退化到零样本）
    #   alpha=0.2 → 保留 80% 原始特征 + 20% Adapter 特征（论文推荐）
    #   alpha=0.5 → 原始特征和 Adapter 特征各占一半
    #   alpha=1.0 → 完全使用 Adapter 特征（可能丢失预训练知识）
    adapter_alpha = 0.2

    # adapter_epochs : int — CLIP-Adapter 训练的最大轮数
    adapter_epochs = 50

    # adapter_batch_size : int — CLIP-Adapter 训练的批大小
    adapter_batch_size = 32

    # adapter_lr : float — CLIP-Adapter 的学习率
    #   使用 Adam 优化器时推荐 0.001
    adapter_lr = 0.001

    # adapter_weight_decay : float — 权重衰减系数
    adapter_weight_decay = 1e-4

    # =============================================================
    # 评估设置
    # =============================================================
    # test_batch_size : int — 测试时的批大小
    #   测试时不需要梯度，可以设置较大的批大小加速推理
    #   显存允许的情况下可以增大（如 128 或 256）
    test_batch_size = 64

    # =============================================================
    # 输出路径
    # =============================================================
    # output_dir : str — 所有实验结果和可视化图表的输出目录
    #   目录内容：
    #     outputs/
    #       ├── checkpoints/       # 模型权重文件（.pth）
    #       ├── clip_zero-shot.csv # Zero-shot 结果
    #       ├── coop.csv           # CoOp 结果
    #       ├── clip-adapter.csv   # CLIP-Adapter 结果
    #       ├── accuracy_curve.png # 准确率对比图
    #       ├── training_time.png  # 训练时间对比图
    #       └── gpu_memory.png     # 显存占用对比图
    output_dir = './outputs'

    # results_file : str — （已废弃）旧版单文件结果路径
    #   现改为每个模型独立 CSV 文件，此字段保留仅用于兼容
    results_file = './outputs/results.csv'
