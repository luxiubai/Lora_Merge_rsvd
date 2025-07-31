# Lora_Merge_rsvd

本项目是基于随机 SVD（rSVD）的 LoRA 合并 方法，核心目标是探索在保持模型性能的同时，合并 LoRA 并有效降低合并后 LoRA 模型的秩和参数以及合并耗时。

## 核心功能

### 1. 基于随机 SVD 的 LoRA 合并 (`merge_rsvd.py`)

`merge_rsvd.py` 实现了基于随机 SVD 的 LoRA 合并算法。该方法旨在通过低秩近似来合并两个 LoRA 适配器，从而在合并过程中动态调整秩，以优化参数效率和性能。

**主要特点:**

- **随机 SVD (rSVD)**: 利用随机 SVD 算法高效地对合并后的权重矩阵进行低秩分解。
- **动态秩调整**: 根据能量阈值和预设的最小/最大秩自动确定合并后的 LoRA 模块的有效秩。
- **参数效率**: 旨在生成更小、更高效的 LoRA 适配器，减少存储和推理成本。
- **性能保留**: 通过特征保留机制，尽可能保持原始 LoRA 模型的性能。

**使用示例 (在 `merge_rsvd.py` 中):**

```python
lora_path1="/lora/PIQA"
lora_path1="/lora/SIQA"
output_directory="lora"
merge_lora_models(
        lora_path1=lora_path1,
        lora_path2=lora_path2,
        output_filename=output_filename,
        up_weight_suffix=".lora_B.weight",
        down_weight_suffix=".lora_A.weight",
        alpha_suffix=".alpha",
        epsilon=1e-12,
        min_rank=4,
        max_rank=32,
        energy_threshold=0.99
    )
```

min_rank=max_rank=N 时，生成固定统一秩的 LoRA

### 2. 常规 LoRA 合并方法 (`merge_covention.py`)

`merge_covention.py` 提供了多种常规的 LoRA 合并算法，用于与基于 rSVD 的方法进行对比。

**支持的合并方法:**

- **M_Concatenate (拼接合并)**: 直接拼接两个 LoRA 的 Up 和 Down 矩阵，导致秩的叠加。
- **M_Geometric_mean (几何平均合并)**: 对 LoRA 权重进行几何平均，通常用于保持权重分布的平衡。
- **M_SVD (SVD 合并)**: 将两个 LoRA 的权重矩阵相加后，再进行 SVD 分解以获得新的低秩 LoRA。
- **M_Weighted_avg (加权平均合并)**: 对两个 LoRA 的权重矩阵进行加权平均。

**使用示例 (在 `merge_covention.py` 中):**

```python
lora_path1="/lora/PIQA"
lora_path1="/lora/SIQA"
output_directory="lora"
merge_methods_main = {
    "M_Concatenate": concatenate_merge,
    "M_Geometric_mean": geometric_mean_merge,
    "M_SVD": svd_merge,
    "M_Weighted_avg": weighted_merge
    }

merge_lora_adapters_covention(
    lora_path1=lora_path1,
    lora_path2=lora_path2,
    output_dir=output_directory,
    merge_methods=merge_methods_main,
    up_weight_suffix=".lora_B.weight",
    down_weight_suffix=".lora_A.weight",
    alpha_suffix=".alpha",
    merge_weight=0.5
    )
```

### Stable Diffusion XL 键名

如需合并 Stable Diffusion XL 的 LoRA，请更改键名匹配，其余架构的模型同理

```python
up_weight_suffix = ".lora_up.weight"
down_weight_suffix = ".lora_down.weight"
alpha_suffix = ".alpha"
```

## 实验报告

项目通过实验对比了不同 LoRA 合并方法的性能和效率。

### LoRA 合并实验报告 (`report/lora_merge_report.md`)

| 方法             | 参数总量  | 平均秩 | 合并耗时(s) |
| ---------------- | --------- | ------ | ----------- |
| M_Concatenate    | 2,293,760 | 16.00  | 0.06        |
| M_Geometric_mean | 1,146,880 | 8.00   | 0.08        |
| M_SVD            | 1,146,880 | 8.00   | 15.64       |
| M_Weighted_avg   | 1,146,880 | 8.00   | 15.66       |
| M_rSVD_strategy1 | 1,003,520 | 7.00   | 0.5253      |
| M_rSVD_strategy2 | 1,146,880 | 8.00   | 0.2981      |
| M_rSVD_strategy3 | 2,293,760 | 16.00  | 0.3089      |

从报告中可以看出，基于 rSVD 的合并策略（如 `M_rSVD_strategy1`）在参数总量和平均秩上表现出优势，同时合并速度也相对较快。

### LoRA 评估报告 (`report/lora_evaluation_report.md`)

该报告展示了不同合并模型在 PIQA 和 SIQA 任务上的准确率，以及评估时间和可训练参数量。

| 模型类型              | PIQA 准确率 | SIQA 准确率 | 评估时间 | 可训练参数  |
| --------------------- | ----------- | ----------- | -------- | ----------- |
| 基础模型              | 66.38%      | 42.63%      | 234.6s   | 596,049,920 |
| PIQA LoRA             | 69.26%      | 47.29%      | 293.2s   | 4,587,520   |
| SIQA LoRA             | 67.52%      | 52.46%      | 294.4s   | 4,587,520   |
| M_Concatenate LoRA    | 69.37%      | 50.87%      | 297.2s   | 9,175,040   |
| M_Geometric_mean LoRA | 66.16%      | 42.99%      | 294.1s   | 4,587,520   |
| M_SVD LoRA            | 66.41%      | 43.40%      | 293.2s   | 4,587,520   |
| M_Weighted_avg LoRA   | 66.48%      | 43.30%      | 293.5s   | 4,587,520   |
| M_rSVD1 LoRA          | 69.75%      | 50.51%      | 293.8s   | 4,257,792   |
| M_rSVD2 LoRA          | 69.42%      | 50.51%      | 293.2s   | 4,587,520   |
| M_rSVD3 LoRA          | 68.39%      | 49.33%      | 294.2s   | 9,175,040   |

```
rSVD1:'min_rank': 4, 'max_rank': 32, 'energy_threshold': 0.99
rSVD2:'min_rank': 32, 'max_rank': 32, 'energy_threshold': 1.0
rSVD3:'min_rank': 64, 'max_rank': 64, 'energy_threshold': 1.0
```

评估报告显示，`M_rSVD1 LoRA` 在 PIQA 和 SIQA 两项任务中都能达到不错的成绩，同时其可训练参数量也相对较低，这表明 rSVD 合并方法在效果和效率之间取得了良好的平衡。

## 安装与使用

### 环境准备

本项目使用 `uv` 作为包管理器。请确保您的环境中已安装 `uv`。

```bash
pip install uv
```

### 安装依赖

```bash
uv sync
```

**注意**: 如果 Pytorch 与设备需求不一致需要更改 pyproject.toml

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }
torchaudio = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

### 按需运行脚本

```bash
uv download.py  #下载模型
uv train.py #训练LoRA
uv run merge_rsvd.py    #随机SVD合并
uv run merge_covention.py   #常规合并
uv run evaluate.py  #性能评估
```

## 算法公式

定义以下符号：

- $U_1, D_1$ 分别代表 LoRA 模型 A 的 `lora_up` 和 `lora_down` 矩阵。
- $U_2, D_2$ 分别代表 LoRA 模型 B 的 `lora_up` 和 `lora_down` 矩阵。
- $\alpha_1, r_1$ 分别代表 LoRA 模型 A 的 `alpha` 和 `rank`。
- $\alpha_2, r_2$ 分别代表 LoRA 模型 B 的 `alpha` 和 `rank`。
- $W_1 = U_1 D_1 \cdot (\alpha_1 / r_1)$ 代表 LoRA 模型 A 的权重矩阵。
- $W_2 = U_2 D_2 \cdot (\alpha_2 / r_2)$ 代表 LoRA 模型 B 的权重矩阵。
- $w$ 代表加权合并中的权重因子。

### 1. 基于随机 SVD 的合并 (rSVD Merge)

```math
W_{avg} = w \cdot W_1 + (1-w) \cdot \sqrt{\frac{\alpha_2}{\alpha_1}} \cdot W_2
```

```math
U_{new}, S_{new}, V_{new}^T = \text{rSVD}(W_{avg}, k)
```

```math
U'_{new} = U_{new} \cdot \sqrt{S_{new}}, \quad D'_{new} = \sqrt{S_{new}} \cdot V_{new}^T
```

### 2. 拼接合并 (Concatenate Merge)

```math
U_{new} = \text{concat}(U_1, U_2, \text{dim}=1)
```

```math
D_{new} = \text{concat}(D_1, D_2, \text{dim}=0)
```

### 3. 几何平均合并 (Geometric Mean Merge)

```math
U_{new} = \text{sign}(U_1 \odot U_2) \odot \sqrt{|U_1 \odot U_2|}
```

```math
D_{new} = \text{sign}(D_1 \odot D_2) \odot \sqrt{|D_1 \odot D_2|}
```

### 4. SVD 合并 (SVD Merge)

```math
W_{merged} = W_1 + W_2
```

```math
U_{new}, S_{new}, V_{new}^T = \text{SVD}(W_{merged}, k)
```

```math
U'_{new} = U_{new} \cdot \text{diag}(\sqrt{S_{new}}), \quad D'_{new} = \text{diag}(\sqrt{S_{new}}) \cdot V_{new}^T
```

### 5. 加权平均合并 (Weighted Average Merge)

```math
W_{merged} = w \cdot W_1 + (1-w) \cdot W_2
```

```math
U_{new}, S_{new}, V_{new}^T = \text{SVD}(W_{merged}, k)
```

```math
U'_{new} = U_{new} \cdot \text{diag}(\sqrt{S_{new}}), \quad D'_{new} = \text{diag}(\sqrt{S_{new}}) \cdot V_{new}^T
```
