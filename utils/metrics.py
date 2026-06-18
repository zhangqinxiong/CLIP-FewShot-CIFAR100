"""
=====================================================================
  工具函数模块 —— 推理时间和 GPU 显存测量
=====================================================================

功能说明：
  本模块提供实验评估中常用的两个辅助函数：
    1. measure_inference_time() — 精确测量模型单张图片的推理时间
    2. get_gpu_memory()         — 获取 GPU 的峰值显存使用量

这些指标对于全面评估模型性能至关重要：
  - 准确率（acc）衡量模型效果
  - 推理时间衡量模型效率（部署友好性）
  - GPU 显存衡量模型资源需求（硬件成本）
"""

import torch      # PyTorch：CUDA 同步、显存查询
import time       # 高精度计时
import numpy as np  # 计算统计量


@torch.no_grad()
def measure_inference_time(model, sample_input, class_names, num_runs=50, warmup=10):
    """
    测量模型对单张图片推理的平均耗时。

    为什么需要测量推理时间？
      推理时间（latency）是模型部署的重要指标，直接影响用户体验。
      对于实时应用（如视频监控、交互式系统），推理时间尤为关键。

    测量方法：
      1. Warmup 阶段：
         - 执行 warmup 轮推理（不计时）
         - 目的：让 GPU 完成初始化（kernel 加载、缓存预热等），
                 达到稳定运行状态，避免首次推理的冷启动开销
      2. 计时阶段：
         - 执行 num_runs 轮推理并计时
         - 取平均值作为单张图片的推理时间
         - 使用 torch.cuda.synchronize() 确保 GPU 操作完成再计时

    影响因素：
      - backbone 选择：ViT-B/32 < RN50 < ViT-B/16 < ViT-L/14
      - 类别数量：文本特征构建时间随类别数线性增长
      - 批次大小：通常 batch_size=1 用于测量实际推理延迟
      - 硬件：GPU 型号、CPU 频率、内存带宽

    参数详解：
        model        : object — 待测模型，需支持 predict() 或 forward()
        sample_input : torch.Tensor — 单张测试图片 [1, 3, 224, 224]
        class_names  : list[str] — 类别名称列表，用于预测
        num_runs     : int  — 计时轮次，默认 50
            更多轮次 → 统计更稳定 → 耗时更长
        warmup       : int  — 预热轮次，默认 10
            更多预热 → GPU 更稳定 → 测量更准确

    返回值：
        float — 平均推理时间（毫秒，ms），保留浮点精度

    使用示例：
        sample = torch.randn(1, 3, 224, 224).to('cuda')
        infer_time = measure_inference_time(model, sample, class_names)
        print(f'Average inference time: {infer_time:.2f} ms')
    """
    # 确定模型所在设备
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device

    # 内部包装函数：统一不同模型的调用接口
    # 优先使用 predict() 方法，否则使用 forward()
    def run_model():
        try:
            # 尝试 model.predict(images, class_names) 接口
            if hasattr(model, 'predict'):
                model.predict(sample_input, class_names)
            else:
                # 尝试 model(images) 接口（某些模型不需要 class_names）
                model(sample_input)
        except TypeError:
            # 如果上述调用失败，尝试 model(images, class_names) 接口
            model(sample_input, class_names)

    # ===== Warmup 预热阶段 =====
    # GPU 初始化滞后：第一次调用 GPU 时，需要加载 CUDA kernel、
    # 分配显存等操作，速度较慢。预热后再计时可得到真实的稳态性能。
    for _ in range(warmup):
        run_model()

    # ===== 计时阶段 =====
    # torch.cuda.synchronize()：等待 GPU 完成所有操作
    # 没有这行，time.time() 可能在 GPU 真正完成前就返回
    torch.cuda.synchronize()
    start = time.time()

    for _ in range(num_runs):
        run_model()

    # 再次同步，确保 GPU 全部计算完成
    torch.cuda.synchronize()
    elapsed = time.time() - start  # 总耗时（秒）

    # 转换为毫秒并计算平均值
    avg_time = elapsed / num_runs * 1000  # 毫秒
    return avg_time


def get_gpu_memory(device='cuda'):
    """
    获取 GPU 的峰值显存使用量。

    为什么需要测量显存？
      显存占用决定了模型能否在特定 GPU 上运行。
      对于模型部署和硬件选型至关重要。

    测量方法：
      使用 torch.cuda.max_memory_allocated() 获取从程序开始
      （或上次 reset_peak_memory_stats() 以来）的峰值显存分配量。

    使用注意事项：
      在测量前需要调用：
        torch.cuda.reset_peak_memory_stats()  # 重置峰值统计
        torch.cuda.empty_cache()              # 清空缓存
      这样可以获得干净的、只包含当前操作的显存占用数据。

    参数说明：
        device : str — GPU 设备名称，默认 'cuda'

    返回值（tuple）：
        allocated : float — 峰值分配的显存量（GB）
            实际使用的显存，反映模型参数、中间激活的占用
        reserved  : float — 峰值保留的显存量（GB）
            PyTorch 的缓存分配器保留的显存（包含 allocated + 碎片）
            通常略大于 allocated

    使用示例：
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        # ... 执行模型操作 ...
        alloc, reserv = get_gpu_memory()
        print(f'Peak memory: {alloc:.2f} GB (allocated), {reserv:.2f} GB (reserved)')
    """
    if not torch.cuda.is_available():
        return 0, 0  # 没有 GPU 返回 0

    # 转换为 GB（1024^3 字节 = 1 GB）
    allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    return allocated, reserved
