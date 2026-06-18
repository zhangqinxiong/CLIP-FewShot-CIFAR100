"""
=============================================================
  主训练脚本 —— CLIP Few-shot 图像分类基准实验入口
=============================================================

功能说明：
  本脚本是项目的核心入口，负责依次运行 3 种方法的实验：
    1. CLIP Zero-shot  —— 零样本基线（无训练，直接评估 CLIP 原始分类能力）
    2. CoOp            —— 可学习 Prompt 向量（Context Optimization）
    3. CLIP-Adapter    —— 特征适配器微调

  每种方法在 5 个 few-shot 设置下评估：1-shot, 2-shot, 4-shot, 8-shot, 16-shot。
  每个设置代表每个类别仅使用 shot 张标注图片进行训练。

自动化管理：
  - 权重检查点（checkpoint）自动保存和加载，已训练的设置自动跳过
  - 每个模型的结果独立保存为一个 CSV 文件
  - 实时保存结果到磁盘，训练中断不会丢失历史数据
  - 自动创建输出目录结构

输出文件结构（均在 outputs/ 下）：
  checkpoints/         模型权重文件目录
    ├── clip_zero-shot_shot1.pth
    ├── coop_shot1.pth
    └── clip-adapter_shot1.pth ...
  clip_zero-shot.csv   零样本结果
  coop.csv             CoOp 结果
  clip-adapter.csv     CLIP-Adapter 结果
"""

import os          # 文件和路径操作：创建目录、拼接路径
import sys         # 系统相关：修改 Python 模块搜索路径
import torch       # PyTorch 深度学习框架：张量计算、权重保存/加载
import time        # 时间模块：测量训练和推理耗时
import csv         # CSV 文件操作：将实验结果写入 CSV 文件
import warnings    # 警告控制：忽略不必要的警告信息
warnings.filterwarnings('ignore')  # 忽略所有警告，保持输出整洁

# ============ 路径配置 ============
# 将项目根目录添加到 sys.path，确保 Python 能正确导入项目内的模块
# os.path.abspath(__file__) 获取当前脚本的绝对路径
# os.path.dirname()         获取目录部分（去掉文件名）
# sys.path.insert(0, ...)    将项目根目录插入搜索路径首位，优先查找
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ============ 导入项目内部模块 ============
from configs.config import Config                           # 全局配置类：集中管理所有超参数
from datasets import build_cifar100_fewshot, CIFAR100_CLASSES  # 数据集构建函数和类别列表
from models import CLIPZeroShot, CoOp, CLIPAdapter          # 三种模型类
from models.coop import train_coop, evaluate_coop           # CoOp 的训练和评估函数
from models.clip_adapter import train_adapter, evaluate_adapter  # CLIP-Adapter 的训练和评估函数
from utils.metrics import measure_inference_time, get_gpu_memory  # 工具函数：推理时间测量和 GPU 显存监控
from torch.utils.data import DataLoader                     # PyTorch 数据加载器：批量加载数据

# ============ 初始化全局配置 ============
config = Config()                             # 实例化配置对象，读取 configs/config.py 中的所有参数
os.makedirs(config.output_dir, exist_ok=True) # 创建输出目录（如果不存在），exist_ok=True 避免已存在时报错

# 权重文件存储目录（checkpoints），用于保存训练好的模型参数
CKPT_DIR = os.path.join(config.output_dir, 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)          # 创建权重目录

# 全局结果列表：累积所有实验的结果，用于后续保存到 CSV
results = []


def log_result(model_name, shot, acc, train_time=0.0, infer_time=0.0, gpu_mem=0.0):
    """
    记录并保存单个实验结果。

    功能流程：
    1. 将实验结果格式化为字典，添加到全局 results 列表
    2. 调用 save_model_result() 立即写入对应的 CSV 文件（实时保存，防丢失）
    3. 在控制台打印该条结果，方便实时监控

    参数说明：
        model_name : str      — 模型名称，如 "CLIP Zero-shot"、"CoOp"、"CLIP-Adapter"
        shot       : int/str  — few-shot 设置，如 1, 2, 4, 8, 16
        acc        : float    — 分类准确率（百分比，如 75.32 表示 75.32%）
        train_time : float    — 训练耗时（秒），零样本无训练为 0
        infer_time : float    — 单张图片平均推理时间（毫秒）
        gpu_mem    : float    — GPU 峰值显存占用（GB）

    数据精度控制：
        acc         → 保留 2 位小数
        train_time  → 保留 2 位小数
        infer_time  → 保留 2 位小数
        gpu_mem     → 保留 3 位小数
    """
    results.append({
        'model': model_name,              # 模型名称
        'shot': shot,                     # few-shot 设置
        'acc': round(acc, 2),             # 准确率，保留 2 位小数
        'train_time_sec': round(train_time, 2),   # 训练时间（秒），保留 2 位小数
        'infer_time_ms': round(infer_time, 2),    # 推理时间（毫秒），保留 2 位小数
        'gpu_mem_gb': round(gpu_mem, 3),          # GPU 显存（GB），保留 3 位小数
    })
    save_model_result(model_name, results)  # 实时写入 CSV 文件，避免中断丢失
    # 控制台输出格式化的结果字符串
    print(f'  >>> {model_name} {shot}-shot: Acc={acc:.2f}% | Train={train_time:.1f}s | Infer={infer_time:.1f}ms | GPU={gpu_mem:.2f}GB')


def save_model_result(model_name, all_results):
    """
    将某个模型的所有结果写入独立的 CSV 文件。

    设计意图：
        每个模型拥有独立的 CSV 文件，便于单独查看和管理。
        文件名由模型名称转换而来（空格→下划线，全小写）。

    文件命名示例：
        "CLIP Zero-shot" → "clip_zero-shot.csv"
        "CoOp"           → "coop.csv"
        "CLIP-Adapter"   → "clip-adapter.csv"

    参数说明：
        model_name  : str — 模型名称，用于确定文件名和筛选结果
        all_results : list — 全局结果列表，从中筛选出属于该模型的所有行

    CSV 列说明：
        model           模型名称
        shot            few-shot 设置
        acc             准确率（%）
        train_time_sec  训练时间（秒）
        infer_time_ms   推理时间（毫秒）
        gpu_mem_gb      GPU 显存占用（GB）
    """
    # 将模型名称转换为安全的文件名：空格替换为下划线，转为小写
    safe_name = model_name.replace(' ', '_').lower()
    # CSV 文件路径：outputs/{safe_name}.csv
    csv_path = os.path.join(config.output_dir, f'{safe_name}.csv')
    # 从全局结果中筛选出当前模型的所有行
    model_rows = [r for r in all_results if r['model'] == model_name]
    # 写入 CSV 文件（使用 w 模式，覆盖写入确保是最新数据）
    with open(csv_path, 'w', newline='') as f:
        # 定义 CSV 列名（固定顺序）
        fieldnames = ['model', 'shot', 'acc', 'train_time_sec', 'infer_time_ms', 'gpu_mem_gb']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()    # 写入表头行
        writer.writerows(model_rows)  # 写入所有数据行


def ckpt_path(model_name, shot):
    """
    计算模型权重文件（checkpoint）的存储路径。

    权重文件命名规范：
        {safe_model_name}_shot{shot}.pth

    示例：
        "CoOp" + 1-shot  → "outputs/checkpoints/coop_shot1.pth"
        "CLIP-Adapter" + 16-shot → "outputs/checkpoints/clip-adapter_shot16.pth"

    参数说明：
        model_name : str — 模型名称，用于生成文件名
        shot       : int — few-shot 设置，用于区分不同 shot 的权重

    返回值：
        str — 权重文件的完整路径
    """
    safe = model_name.replace(' ', '_').lower()  # 模型名称转为安全文件名格式
    return os.path.join(CKPT_DIR, f'{safe}_shot{shot}.pth')


def save_ckpt(state_dict, model_name, shot):
    """
    将模型权重保存到磁盘（checkpoint）。

    保存时机：
        在 CoOp 或 CLIP-Adapter 训练完成后立即保存，
        下次运行时可以直接加载权重，跳过训练。

    参数说明：
        state_dict : dict — 模型的状态字典（通过 model.state_dict() 获取）
        model_name : str  — 模型名称
        shot       : int  — few-shot 设置
    """
    path = ckpt_path(model_name, shot)      # 计算路径
    torch.save(state_dict, path)            # 调用 PyTorch 保存函数
    print(f'  [saved checkpoint] {path}')   # 打印保存路径


def load_ckpt(model_name, shot):
    """
    从磁盘加载模型权重（checkpoint）。

    功能说明：
        检查权重文件是否存在：
        - 如果存在：加载并返回权重字典
        - 如果不存在：返回 None，触发后续训练

    参数说明：
        model_name : str — 模型名称
        shot       : int — few-shot 设置

    返回值：
        dict 或 None — 加载的权重字典，或权重不存在时返回 None
    """
    path = ckpt_path(model_name, shot)  # 计算权重文件路径
    if os.path.exists(path):            # 检查文件是否存在
        print(f'  [loaded checkpoint] {path}')
        # map_location='cpu'：先将权重加载到 CPU，避免 GPU 显存占用
        return torch.load(path, map_location='cpu')
    return None  # 权重不存在，返回 None


# =============================================================
# 实验 1：CLIP Zero-shot 零样本分类
# =============================================================
# CLIP 零样本分类是最简单的基线方法。
# 它完全不使用任何训练数据，直接通过 CLIP 的图文匹配能力进行分类：
#   1. 将 CIFAR-100 的 100 个类别名称通过模板 "a photo of a {class}" 编码为文本特征
#   2. 将测试图片编码为图像特征
#   3. 计算图像与所有类别文本特征的余弦相似度
#   4. 选择相似度最高的类别作为预测结果
#
# 特点：不需要训练，速度最快，是所有 few-shot 方法的性能下限（baseline）
# =============================================================
def run_zeroshot():
    """执行 CLIP Zero-shot 实验：在 1/2/4/8/16 shot 下分别评估"""
    print('\n' + '='*60)
    print('Experiment 1: CLIP Zero-shot')
    print('='*60)

    # 初始化 CLIP 零样本模型（只加载一次模型，所有 shot 共享）
    model = CLIPZeroShot(config.backbone, config.device)

    # 遍历所有 few-shot 设置：1, 2, 4, 8, 16
    for shot in config.shots:
        # 跳过已经完成的设置（防止重复运行）
        if any(r['model']=='CLIP Zero-shot' and r['shot']==str(shot) for r in results):
            print(f'  skip CLIP Zero-shot {shot}-shot (already done)')
            continue

        print(f'\n--- {shot}-shot setting ---')

        # 构建数据集：返回训练集（每类 shot 张）和完整测试集（10000 张）
        # 零样本只用测试集，不需要训练集
        _, test_set = build_cifar100_fewshot(shot=shot, seed=42, data_root=config.data_root)
        # 创建测试数据加载器：批量加载测试图片
        test_loader = DataLoader(test_set, batch_size=config.test_batch_size, num_workers=4)

        # 重置 GPU 显存统计，得到干净的峰值显存测量结果
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()  # 清空 GPU 缓存，保证测量准确性

        # 计时：评估过程（包括特征提取 + 相似度计算）
        start = time.time()
        acc = model.evaluate(test_loader, CIFAR100_CLASSES)  # 评估准确率
        train_time = time.time() - start  # 零样本没有训练，这个时间只是评估时间

        # 测量推理时间：创建一张随机测试图片，测量 20 次推理的平均时间
        sample = torch.randn(1, 3, 224, 224).to(model.device)
        infer_time = measure_inference_time(model, sample, CIFAR100_CLASSES, num_runs=20, warmup=5)

        # 获取 GPU 显存使用量
        gpu_alloc, _ = get_gpu_memory()

        # 记录结果
        log_result('CLIP Zero-shot', shot, acc, train_time, infer_time, gpu_alloc)


# =============================================================
# 实验 2：CoOp (Context Optimization) —— Prompt 学习
# =============================================================
# CoOp 的核心思想是用可学习的连续向量替代人工设计的 prompt 模板。
#
# 原始 CLIP prompt:  "a photo of a {class_name}"  → 固定的文本嵌入
# CoOp prompt:        [V1] [V2] ... [V16] {class_name}  → V 是可学习的
#
# 训练策略：
#   - 冻结 CLIP 文本编码器和图像编码器的全部参数
#   - 仅优化 16 个连续向量（约 8K 参数 = 16 × 512）
#   - 使用 SGD + 动量优化器，余弦退火学习率
#
# 原理说明：
#   人工设计的 "a photo of a" 未必是最优 prompt，不同数据集可能需要
#   不同的表达方式。CoOp 通过梯度下降自动发现最适合当前数据集的 prompt。
# =============================================================
def run_coop():
    """执行 CoOp 实验：在 1/2/4/8/16 shot 下分别训练和评估"""
    print('\n' + '='*60)
    print('Experiment 2: CoOp (Prompt Learning)')
    print('='*60)

    # 遍历所有 few-shot 设置
    for shot in config.shots:
        # 跳过已经完成的设置
        if any(r['model']=='CoOp' and r['shot']==str(shot) for r in results):
            print(f'  skip CoOp {shot}-shot (already done)')
            continue

        print(f'\n--- {shot}-shot setting ---')

        # 构建 few-shot 数据集：每类 shot 张训练 + 完整 10000 张测试
        train_set, test_set = build_cifar100_fewshot(shot=shot, seed=42, data_root=config.data_root)
        # 创建训练和测试数据加载器
        train_loader = DataLoader(train_set, batch_size=config.coop_batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=config.test_batch_size, num_workers=4)

        # 初始化 CoOp 模型
        model = CoOp(config.backbone, config.coop_n_ctx, False, config.device)

        # 初始化训练时间（后续根据是否加载 checkpoint 决定是否更新）
        train_time = 0.0

        # 尝试加载已有权重（支持断点续训）
        ckpt = load_ckpt('CoOp', shot)
        if ckpt is not None:
            # 权重存在：直接将学好的 prompt 向量加载到模型
            model.ctx.data = ckpt['ctx'].to(model.device)
            print(f'  [loaded ctx weights]')
        else:
            # 权重不存在：执行训练
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            start = time.time()
            # train_coop 返回：最佳准确率 + 训练好的 prompt 向量
            best_acc, ctx_state = train_coop(model, train_loader, test_loader, CIFAR100_CLASSES, config)
            train_time = time.time() - start

            # 保存训练好的 prompt 向量到 checkpoint
            save_ckpt({'ctx': ctx_state}, 'CoOp', shot)
            # 将 prompt 向量加载到模型
            model.ctx.data = ctx_state.to(model.device)

        # 评估模型（使用加载的或刚训练好的权重）
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        acc = evaluate_coop(model, test_loader, CIFAR100_CLASSES)

        # 测量推理时间和 GPU 显存
        sample = torch.randn(1, 3, 224, 224).to(model.device)
        infer_time = measure_inference_time(model, sample, CIFAR100_CLASSES, num_runs=20, warmup=5)
        gpu_alloc, _ = get_gpu_memory()

        # 记录结果（如果使用了 checkpoint，训练时间记为 0）
        log_result('CoOp', shot, acc, 0 if ckpt else train_time, infer_time, gpu_alloc)


# =============================================================
# 实验 3：CLIP-Adapter —— 特征适配器微调
# =============================================================
# CLIP-Adapter 在 CLIP 的图像编码器和文本编码器后各添加一个轻量 Adapter。
#
# Adapter 结构（瓶颈结构）：
#   Linear(feat_dim → reduce_dim) → ReLU → Linear(reduce_dim → feat_dim)
#
# 残差连接（关键设计）：
#   output = alpha * adapter(x) + (1 - alpha) * x
#   其中 alpha=0.2，即 80% 保留原始 CLIP 知识，20% 使用任务特定特征
#
# 训练策略：
#   - 冻结 CLIP 全部参数
#   - 仅训练两个 Adapter 模块（约 524K 参数 = 2 × 512×256 × 2）
#   - 使用 Adam 优化器，余弦退火学习率
#
# 与 CoOp 的区别：
#   CoOp 修改输入（prompt），Adapter 修改特征（representation），
#   两者从不同层面适应下游任务。
# =============================================================
def run_adapter():
    """执行 CLIP-Adapter 实验：在 1/2/4/8/16 shot 下分别训练和评估"""
    print('\n' + '='*60)
    print('Experiment 3: CLIP-Adapter')
    print('='*60)

    # 遍历所有 few-shot 设置
    for shot in config.shots:
        # 跳过已经完成的设置
        if any(r['model']=='CLIP-Adapter' and r['shot']==str(shot) for r in results):
            print(f'  skip CLIP-Adapter {shot}-shot (already done)')
            continue

        print(f'\n--- {shot}-shot setting ---')

        # 构建 few-shot 数据集
        train_set, test_set = build_cifar100_fewshot(shot=shot, seed=42, data_root=config.data_root)
        train_loader = DataLoader(train_set, batch_size=config.adapter_batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_set, batch_size=config.test_batch_size, num_workers=4)

        # 初始化 CLIP-Adapter 模型
        model = CLIPAdapter(config.backbone, config.adapter_reduce_dim, config.adapter_alpha, config.device)

        # 初始化训练时间（后续根据是否加载 checkpoint 决定是否更新）
        train_time = 0.0

        # 尝试加载已有权重（支持断点续训）
        ckpt = load_ckpt('CLIP-Adapter', shot)
        if ckpt is not None:
            # 权重存在：直接加载 Adapter 权重
            # strict=False：只加载匹配的 key，不匹配的忽略（CLIP 原始参数不保存）
            model.load_state_dict(ckpt, strict=False)
            print(f'  [loaded adapter weights]')
        else:
            # 权重不存在：执行训练
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            start = time.time()
            # train_adapter 返回：最佳准确率 + Adapter 权重字典
            best_acc, state = train_adapter(model, train_loader, test_loader, CIFAR100_CLASSES, config)
            train_time = time.time() - start

            # 保存 Adapter 权重到 checkpoint
            save_ckpt(state, 'CLIP-Adapter', shot)

        # 评估模型
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        acc = evaluate_adapter(model, test_loader, CIFAR100_CLASSES)

        # 测量推理时间和 GPU 显存
        sample = torch.randn(1, 3, 224, 224).to(model.device)
        infer_time = measure_inference_time(model, sample, CIFAR100_CLASSES, num_runs=20, warmup=5)
        gpu_alloc, _ = get_gpu_memory()

        # 记录结果
        log_result('CLIP-Adapter', shot, acc, 0 if ckpt else train_time, infer_time, gpu_alloc)


def save_results():
    """打印最终提示信息，告知用户结果存储位置"""
    print(f'\nAll results saved to {config.output_dir}/')


# =============================================================
# 程序入口
# =============================================================
if __name__ == '__main__':
    """
    主程序入口：依次运行 3 个实验。

    执行顺序：
    1. 打印实验配置信息
    2. CLIP Zero-shot  （基线）
    3. CoOp （Prompt 学习）
    4. CLIP-Adapter （特征适配器）
    5. 打印完成提示
    """
    print('='*60)
    print('CLIP Few-shot Image Classification Benchmark')
    print('='*60)
    print(f'Backbone: {config.backbone}')      # CLIP 主干网络
    print(f'Device: {config.device}')           # 计算设备
    print(f'Dataset: CIFAR-100')                # 数据集名称
    print(f'Few-shot settings: {config.shots}') # few-shot 设置列表

    # 按顺序执行三个实验
    run_zeroshot()   # 实验 1：零样本基线
    run_coop()       # 实验 2：CoOp
    run_adapter()    # 实验 3：CLIP-Adapter
    save_results()   # 打印完成信息

    print('\n' + '='*60)
    print('All experiments completed!')
    print('='*60)
