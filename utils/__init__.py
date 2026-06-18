"""
utils 包 —— 工具函数模块的统一导出入口。

使用方式：
    from utils.metrics import measure_inference_time, get_gpu_memory

导出函数：
    - measure_inference_time(model, sample, class_names, num_runs, warmup)
        测量模型平均推理时间（毫秒）
    - get_gpu_memory(device)
        获取 GPU 峰值显存使用量（GB）
"""
from .metrics import measure_inference_time, get_gpu_memory
