"""
=============================================================
  结果对比与可视化脚本
=============================================================

功能说明：
  本脚本用于在 run.py 运行完所有实验后，加载各模型的 CSV 结果文件，
  进行对比分析和可视化，帮助研究者直观评估不同方法的性能差距。

核心功能：
  1. 从 CSV 文件加载所有模型的结果数据
  2. 在控制台打印准确率对比表格和训练时间对比表格
  3. 生成三张高清可视化图表（PNG 格式，150 DPI）：
     - accuracy_curve.png   准确率对比柱状图
     - training_time.png    训练时间对比柱状图
     - gpu_memory.png       GPU 显存占用对比柱状图

图表特性：
  - 统一使用高对比度配色方案，色盲友好
  - 柱状图上直接标注数值，一目了然
  - 所有图表保存到 outputs/ 目录
  - 使用 Agg 后端，无需图形界面，适合服务器环境

使用方式：
  python compare.py

前置条件：
  必须先运行 run.py 生成 CSV 结果文件
"""

import os          # 文件和路径操作
import sys         # 系统相关
import csv         # 读取 CSV 结果文件
import numpy as np # 数值计算：计算平均值等统计量
import matplotlib
# 使用非交互式后端 Agg，不依赖 GUI 显示器，适合服务器/远程环境
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # 绘图库

# 导入项目配置
from configs.config import Config

config = Config()  # 实例化配置对象


def load_results(csv_path):
    """
    从 CSV 文件加载实验结果数据。

    读取流程：
      1. 检查 CSV 文件是否存在
      2. 如果不存在，打印错误提示并返回 None
      3. 如果存在，逐行读取并解析为字典列表

    参数说明：
        csv_path : str — CSV 文件的完整路径

    返回值：
        list 或 None — 成功时返回结果列表（每行为一个 dict），
                       文件不存在时返回 None

    返回的数据结构示例：
        [
            {'model': 'CoOp', 'shot': '1', 'acc': '64.28', ...},
            {'model': 'CoOp', 'shot': '2', 'acc': '66.51', ...},
            ...
        ]
    """
    if not os.path.exists(csv_path):
        print(f'Results file not found: {csv_path}')
        print('Please run run.py first to generate results.')
        return None

    results = []
    with open(csv_path, 'r') as f:
        # DictReader 自动将第一行作为列名，后续行作为值
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
    return results


def print_comparison_table(results):
    """
    在控制台打印各模型在不同 shot 下的准确率对比表格。

    表格格式：
        Model              1-shot     4-shot     16-shot
        ------------------------------------------------
        CLIP Zero-shot     62.15      62.15      62.15
        CoOp               64.28      69.83      75.92
        CLIP-Adapter       65.97      71.26      77.83

    参数说明：
        results : list — 包含所有模型结果的字典列表
    """
    print('\n' + '='*70)
    print('Comparison Table: Accuracy (%)')
    print('='*70)
    # 打印表头：左对齐 20 字符，其余列左对齐 10 字符
    print(f'{"Model":<20} {"1-shot":<10} {"4-shot":<10} {"16-shot":<10}')
    print('-'*50)

    # 定义要对比的三个模型
    model_names = ['CLIP Zero-shot', 'CoOp', 'CLIP-Adapter']
    for model_name in model_names:
        accs = {}  # 字典：shot → accuracy
        for r in results:
            if r['model'] == model_name:
                shot = int(r['shot'])   # shot 转为整数，方便排序
                accs[shot] = r['acc']   # 记录准确率

        # 构建一行输出：模型名称 + 1-shot / 4-shot / 16-shot 的准确率
        row = f'{model_name:<20} '
        for shot in [1, 4, 16]:
            val = accs.get(shot, '-')   # 如果没有该 shot 的数据，显示 '-'
            row += f'{str(val):<10} '
        print(row)

    print('='*70)


def print_training_time_table(results):
    """
    在控制台打印各模型在不同 shot 下的训练时间对比表格。

    参数说明：
        results : list — 包含所有模型结果的字典列表

    注意：
        Zero-shot 的训练时间实际为评估时间（因为它不需要训练），
        CoOp 和 CLIP-Adapter 的训练时间包含完整的训练过程，
        如果是从 checkpoint 加载的权重，训练时间记为 0。
    """
    print('\n' + '='*70)
    print('Training Time (seconds)')
    print('='*70)
    print(f'{"Model":<20} {"1-shot":<10} {"4-shot":<10} {"16-shot":<10}')
    print('-'*50)

    model_names = ['CLIP Zero-shot', 'CoOp', 'CLIP-Adapter']
    for model_name in model_names:
        times = {}  # 字典：shot → training time (seconds)
        for r in results:
            if r['model'] == model_name:
                shot = int(r['shot'])
                times[shot] = r['train_time_sec']

        row = f'{model_name:<20} '
        for shot in [1, 4, 16]:
            val = times.get(shot, '-')
            if val != '-':
                val = f'{float(val):.1f}'  # 保留 1 位小数显示
            row += f'{str(val):<10} '
        print(row)

    print('='*70)


def plot_accuracy_comparison(results):
    """
    绘制各模型在不同 few-shot 设置下的准确率对比柱状图。

    图表说明：
        横轴：few-shot 设置（1-shot, 4-shot, 16-shot）
        纵轴：准确率（%），范围 0~100
        不同颜色的柱子代表不同模型（CLIP Zero-shot / CoOp / CLIP-Adapter）
        柱子上方标注具体数值

    参数说明：
        results : list — 包含所有模型结果的字典列表

    输出文件：
        outputs/accuracy_curve.png（150 DPI）
    """
    # 定义模型和 shot 配置
    model_names = ['CLIP Zero-shot', 'CoOp', 'CLIP-Adapter']
    shots = [1, 4, 16]
    # 高对比度配色方案：蓝色、紫红色、橙色（色盲友好）
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    fig, ax = plt.subplots(figsize=(8, 5))  # 创建 8×5 英寸的画布
    x = np.arange(len(shots))               # x 轴位置：[0, 1, 2]
    width = 0.22                            # 每个柱子的宽度（三个柱子并排）

    # 为每个模型绘制一组柱子
    for i, model_name in enumerate(model_names):
        accs = []
        for shot in shots:
            for r in results:
                if r['model'] == model_name and int(r['shot']) == shot:
                    accs.append(float(r['acc']))
                    break  # 找到就跳出内层循环
            else:
                accs.append(0)  # 找不到对应数据，用 0 占位

        # 绘制柱状图：x + i*width 控制柱子位置偏移
        bars = ax.bar(x + i * width, accs, width, label=model_name, color=colors[i])

        # 在柱子顶部标注准确率数值（只标注大于 0 的值）
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{acc:.1f}', ha='center', va='bottom', fontsize=8)

    # 设置图表标签和样式
    ax.set_xlabel('Few-shot Setting', fontsize=12)            # x 轴标签
    ax.set_ylabel('Accuracy (%)', fontsize=12)                # y 轴标签
    ax.set_title('Accuracy Comparison on CIFAR-100', fontsize=14)  # 标题
    ax.set_xticks(x + width * 1.5)                            # 设置 x 轴刻度位置（居中于三组柱子）
    ax.set_xticklabels([f'{s}-shot' for s in shots])          # x 轴刻度标签
    ax.legend(fontsize=10)                                     # 图例
    ax.grid(axis='y', alpha=0.3)                               # 添加水平网格线（半透明）
    ax.set_ylim(0, 100)                                        # y 轴范围 0~100%

    plt.tight_layout()  # 自动调整布局，防止标签被裁切
    path = os.path.join(config.output_dir, 'accuracy_curve.png')
    plt.savefig(path, dpi=150)  # 以 150 DPI 保存高清图片
    print(f'Accuracy plot saved to {path}')
    plt.close()  # 关闭当前图形，释放内存


def plot_training_time(results):
    """
    绘制各模型在不同 few-shot 设置下的训练时间对比柱状图。

    图表说明：
        横轴：few-shot 设置（1-shot, 4-shot, 16-shot）
        纵轴：时间（秒）
        柱子上方标注训练时间（取整到秒）

    参数说明：
        results : list — 包含所有模型结果的字典列表

    输出文件：
        outputs/training_time.png（150 DPI）
    """
    model_names = ['CLIP Zero-shot', 'CoOp', 'CLIP-Adapter']
    shots = [1, 4, 16]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(shots))
    width = 0.22

    for i, model_name in enumerate(model_names):
        times = []
        for shot in shots:
            for r in results:
                if r['model'] == model_name and int(r['shot']) == shot:
                    val = float(r['train_time_sec'])
                    times.append(val)
                    break
            else:
                times.append(0)

        bars = ax.bar(x + i * width, times, width, label=model_name, color=colors[i])
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{t:.0f}s', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Few-shot Setting', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training Time Comparison', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([f'{s}-shot' for s in shots])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.output_dir, 'training_time.png')
    plt.savefig(path, dpi=150)
    print(f'Training time plot saved to {path}')
    plt.close()


def plot_gpu_memory(results):
    """
    绘制各模型在实验中的 GPU 显存占用对比柱状图。

    图表说明：
        横轴：模型名称（CLIP Zero-shot / CoOp / CLIP-Adapter）
        纵轴：峰值 GPU 显存占用（GB）
        每个柱子的高度表示该模型在所有 shot 设置下的平均显存占用
        柱子上方标注具体数值

    统计方法：
        对每个模型，计算其在所有 shot 下峰值显存占用的平均值。
        这样可以综合反映模型在典型使用场景下的显存需求。

    参数说明：
        results : list — 包含所有模型结果的字典列表

    输出文件：
        outputs/gpu_memory.png（150 DPI）
    """
    model_names = ['CLIP Zero-shot', 'CoOp', 'CLIP-Adapter']
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, model_name in enumerate(model_names):
        # 收集该模型在所有 shot 下的 GPU 显存占用
        mems = []
        for r in results:
            if r['model'] == model_name:
                mems.append(float(r['gpu_mem_gb']))
        if mems:
            avg_mem = np.mean(mems)  # 计算平均值
            ax.bar(model_name, avg_mem, color=colors[i], width=0.5)
            ax.text(i, float(avg_mem) + 0.02, f'{avg_mem:.2f}GB', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Peak GPU Memory (GB)', fontsize=12)
    ax.set_title('GPU Memory Usage', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(config.output_dir, 'gpu_memory.png')
    plt.savefig(path, dpi=150)
    print(f'GPU memory plot saved to {path}')
    plt.close()


# =============================================================
# 程序入口
# =============================================================
if __name__ == '__main__':
    """
    主程序流程：
    1. 从各模型的独立 CSV 文件加载结果
    2. 打印准确率对比表格
    3. 打印训练时间对比表格
    4. 生成三张可视化图表

    结果文件命名规则（由 run.py 生成）：
        CLIP Zero-shot → clip_zero-shot.csv
        CoOp           → coop.csv
        CLIP-Adapter   → clip-adapter.csv
    """
    # 所有模型结果汇总列表
    all_results = []
    # 定义三个模型的 CSV 文件名（由 run.py 的 save_model_result 生成）
    model_files = ['clip_zero-shot.csv', 'coop.csv', 'clip-adapter.csv']

    # 逐文件加载结果
    for fname in model_files:
        path = os.path.join(config.output_dir, fname)
        rows = load_results(path)
        if rows:
            all_results.extend(rows)  # 合并到汇总列表

    # 检查是否有数据
    if not all_results:
        print('No results found. Please run run.py first.')
        sys.exit(1)

    # 打印对比表格
    print_comparison_table(all_results)       # 准确率表格
    print_training_time_table(all_results)    # 训练时间表格

    # 生成可视化图表
    plot_accuracy_comparison(all_results)     # 准确率柱状图
    plot_training_time(all_results)           # 训练时间柱状图
    plot_gpu_memory(all_results)              # GPU 显存柱状图

    print('\nAll plots generated in:', config.output_dir)
