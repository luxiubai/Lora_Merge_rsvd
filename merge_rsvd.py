import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
from typing import Tuple
from torch import svd_lowrank
import time
import os

def _to_device(state_dict, device):
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.to(device)
    return state_dict

class RSVD:
    """随机SVD封装"""
    def __init__(self, target_components: int): 
        self.target_components = target_components

    def apply_rsvd(self, matrix):
        if matrix.numel() == 0:
            return torch.empty(matrix.shape[0], 0), torch.empty(0), torch.empty(0, matrix.shape[1])
            
        U, s, V = svd_lowrank(matrix, q=self.target_components, niter=2)
        
        return U, s, V.mT

def merge_svd_components(
    lora_up1: torch.Tensor, lora_down1: torch.Tensor, 
    lora_up2: torch.Tensor, lora_down2: torch.Tensor,
    alpha1: float, r1: int, alpha2: float, r2: int,
    epsilon: float = 1e-12,
    min_rank: int = 1,
    max_rank: int = 512,
    energy_threshold: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor, int, float]:
    
    original_dtype = lora_up1.dtype
    
    lora_up1 = lora_up1.to(torch.float32)
    lora_down1 = lora_down1.to(torch.float32)
    lora_up2 = lora_up2.to(torch.float32)
    lora_down2 = lora_down2.to(torch.float32)

    scale1 = torch.tensor(alpha1 / max(r1, 1), device=lora_down1.device)
    scale2 = torch.tensor(alpha2 / max(r2, 1), device=lora_down2.device)
    
    N1 = lora_up1 @ (lora_down1 * scale1)
    N2 = lora_up2 @ (lora_down2 * scale2)

    norm_N1 = torch.norm(N1) + epsilon
    norm_N2 = torch.norm(N2) + epsilon
    
    norm_ratio = torch.min(norm_N1, norm_N2) / torch.max(norm_N1, norm_N2)
    if norm_ratio < 0.5:
        weight = torch.tensor(0.5, device=N1.device)
    else:
        weight = norm_N1 / (norm_N1 + norm_N2)
    
    alpha_ratio_val = torch.sqrt(torch.tensor(alpha2 / alpha1, device=N1.device)) if alpha1 > 0 else torch.tensor(1.0, device=N1.device)

    W_avg = weight * N1 + (1.0 - weight) * alpha_ratio_val * N2

    base_rank = int((min(int(r1), int(r2)) * max(int(r1), int(r2)))**0.5)
    
    initial_q = min(
        max(min_rank, int(base_rank * float(energy_threshold))),
        min(W_avg.shape[0], W_avg.shape[1]),
        max_rank
    )
    
    protector = RSVD(target_components=initial_q)
    U, s, Vh = protector.apply_rsvd(W_avg)
    
    total_energy = s.sum() + epsilon
    cumulative = torch.cumsum(s, 0) / total_energy
    effective_rank = torch.searchsorted(cumulative, energy_threshold).item() + 1
    
    effective_rank = min(max_rank, max(min_rank, effective_rank))
    effective_rank = min(effective_rank, s.numel())

    s_sqrt = torch.sqrt(s[:effective_rank] + epsilon)
    lora_down_new = Vh[:effective_rank, :] * s_sqrt[:, None]
    lora_up_new = U[:, :effective_rank] * s_sqrt[None, :]
    
    lora_down_new = lora_down_new.to(original_dtype)
    lora_up_new = lora_up_new.to(original_dtype)

    final_rank = effective_rank
    
    orig_norm = 0.5 * (norm_N1 + norm_N2)
    recon_norm = torch.norm(lora_up_new @ lora_down_new)
    preservation = recon_norm / orig_norm if orig_norm > epsilon else 0.0
    
    if (torch.isnan(lora_down_new).any() or torch.isinf(lora_down_new).any() or
        torch.isnan(lora_up_new).any() or torch.isinf(lora_up_new).any()):
        lora_down_new = lora_down1 * scale1
        lora_up_new = lora_up1
        final_rank = r1
        preservation = 1.0
    
    return lora_down_new, lora_up_new, int(final_rank), float(preservation)

def merge_lora_models(
    lora_path1: str, 
    lora_path2: str, 
    output_filename: str,
    up_weight_suffix: str,
    down_weight_suffix: str,
    alpha_suffix: str,
    epsilon: float = 1e-12,
    min_rank: int = 1,
    max_rank: int = 512,
    energy_threshold: float = 0.95
) -> Tuple[int, float, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    start_time = time.time()

    lora1_state_dict = load_file(lora_path1)
    lora2_state_dict = load_file(lora_path2)

    lora1_state_dict = _to_device(lora1_state_dict, device)
    lora2_state_dict = _to_device(lora2_state_dict, device)

    merged_state_dict = {}
    
    def _get_lora_module_names(state_dict):
        module_names = set()
        for key in state_dict.keys():
            if key.endswith(up_weight_suffix):
                module_names.add(key.replace(up_weight_suffix, ''))
        return module_names

    all_module_names = _get_lora_module_names(lora1_state_dict)
    all_module_names.update(_get_lora_module_names(lora2_state_dict))

    total_params = 0
    total_rank = 0
    merged_modules_count = 0

    for module_name in tqdm(all_module_names, desc="正在合并LoRA模块"):
        lora_up_key = f"{module_name}{up_weight_suffix}"
        lora_down_key = f"{module_name}{down_weight_suffix}"
        alpha_key = f"{module_name}{alpha_suffix}"

        has_lora1 = lora_up_key in lora1_state_dict and lora_down_key in lora1_state_dict
        has_lora2 = lora_up_key in lora2_state_dict and lora_down_key in lora2_state_dict

        if has_lora1 and has_lora2:
            lora_up1 = lora1_state_dict[lora_up_key]
            lora_down1 = lora1_state_dict[lora_down_key]
            lora_up2 = lora2_state_dict[lora_up_key]
            lora_down2 = lora2_state_dict[lora_down_key]

            alpha1_val = lora1_state_dict.get(alpha_key, torch.tensor(lora_down1.shape[0], dtype=torch.float32))
            alpha1 = float(alpha1_val.item()) if isinstance(alpha1_val, torch.Tensor) else float(alpha1_val)
            r1 = int(lora_down1.shape[0])
            
            alpha2_val = lora2_state_dict.get(alpha_key, torch.tensor(lora_down2.shape[0], dtype=torch.float32))
            alpha2 = float(alpha2_val.item()) if isinstance(alpha2_val, torch.Tensor) else float(alpha2_val)
            r2 = int(lora_down2.shape[0])
            
            result = merge_svd_components(
                lora_up1, lora_down1, lora_up2, lora_down2,
                alpha1, r1, alpha2, r2,
                epsilon=epsilon,
                min_rank=min_rank,
                max_rank=max_rank,
                energy_threshold=energy_threshold
            )
            
            if result[0] is not None:
                lora_down_new, lora_up_new, final_rank, preservation = result
                merged_state_dict[lora_down_key] = lora_down_new.contiguous()
                merged_state_dict[lora_up_key] = lora_up_new.contiguous()
                merged_state_dict[alpha_key] = torch.tensor([final_rank], dtype=torch.float32)
                
                total_params += lora_down_new.numel() + lora_up_new.numel()
                total_rank += final_rank
                merged_modules_count += 1

                tqdm.write(f"模块 {module_name} 合并完成，新秩: {final_rank}, 特征保留: {preservation:.4f}")
            else:
                merged_state_dict[lora_up_key] = lora_up1
                merged_state_dict[lora_down_key] = lora_down1
                if alpha_key in lora1_state_dict:
                    merged_state_dict[alpha_key] = lora1_state_dict[alpha_key]
                
                total_params += lora_up1.numel() + lora_down1.numel()
                total_rank += r1
                merged_modules_count += 1

        elif has_lora1:
            merged_state_dict[lora_up_key] = lora1_state_dict[lora_up_key]
            merged_state_dict[lora_down_key] = lora1_state_dict[lora_down_key]
            if alpha_key in lora1_state_dict:
                merged_state_dict[alpha_key] = lora1_state_dict[alpha_key]
            
            total_params += lora1_state_dict[lora_up_key].numel() + lora1_state_dict[lora_down_key].numel()
            total_rank += int(lora1_state_dict[lora_down_key].shape[0])
            merged_modules_count += 1

        elif has_lora2:
            merged_state_dict[lora_up_key] = lora2_state_dict[lora_up_key]
            merged_state_dict[lora_down_key] = lora2_state_dict[lora_down_key]
            if alpha_key in lora2_state_dict:
                merged_state_dict[alpha_key] = lora2_state_dict[alpha_key]
            
            total_params += lora2_state_dict[lora_up_key].numel() + lora2_state_dict[lora_down_key].numel()
            total_rank += int(lora2_state_dict[lora_down_key].shape[0])
            merged_modules_count += 1

    for key in lora1_state_dict:
        if not (key.endswith(up_weight_suffix) or 
                key.endswith(down_weight_suffix) or 
                key.endswith(alpha_suffix)):
            merged_state_dict[key] = lora1_state_dict[key]
    
    for key in lora2_state_dict:
        if not (key.endswith(up_weight_suffix) or 
                key.endswith(down_weight_suffix) or 
                key.endswith(alpha_suffix)) and key not in merged_state_dict:
            merged_state_dict[key] = lora2_state_dict[key]

    save_file(merged_state_dict, output_filename)
    print(f"合并后的LoRA已保存到: {output_filename}")
    
    end_time = time.time()
    merge_speed = end_time - start_time

    average_rank = total_rank / merged_modules_count if merged_modules_count > 0 else 0

    print("合并完成！")
    return total_params, average_rank, merge_speed

def gen_report(results: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "lora_merge_report.md")

    file_exists = os.path.exists(report_path)
    file_is_empty = not file_exists or os.stat(report_path).st_size == 0

    with open(report_path, "a", encoding="utf-8") as f:
        if file_is_empty:
            f.write("# LoRA 合并实验报告\n\n")
            f.write("| 方法 | 参数总量 | 平均秩 | 合并耗时(s) |\n")
            f.write("|------|---------|-------|--------|\n")
        
        for res in results:
            f.write(
                f"| M_rSVD_{res['strategy_name']} | "
                f"{res['total_params']:,} | {res['average_rank']:.2f} | "
                f"{res['merge_speed']:.4f} |\n"
            )
    
    print(f"\n实验报告已保存至: {report_path}")

if __name__ == "__main__":
    lora_path_piqa = "lora/PIQA/adapter_model.safetensors"
    lora_path_siqa = "lora/SIQA/adapter_model.safetensors"

    strategies = [
        {'min_rank': 4, 'max_rank': 32, 'energy_threshold': 0.99, 'name': 'strategy1'},
        {'min_rank': 32, 'max_rank': 32, 'energy_threshold': 1.0, 'name': 'strategy2'},
        {'min_rank': 64, 'max_rank': 64, 'energy_threshold': 1.0, 'name': 'strategy3'}
    ]

    results = []

    for strategy in strategies:
        print(f"\n--- 正在执行合并策略: {strategy['name']} ---")
        output_filename = f"lora/M_rSVD_{strategy['name']}/adapter_model.safetensors"
        
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)

        total_params, average_rank, merge_speed = merge_lora_models(
            lora_path1=lora_path_piqa,
            lora_path2=lora_path_siqa,
            output_filename=output_filename,
            up_weight_suffix=".lora_B.weight",
            down_weight_suffix=".lora_A.weight",
            alpha_suffix=".alpha",
            epsilon=1e-12,
            min_rank=strategy['min_rank'],
            max_rank=strategy['max_rank'],
            energy_threshold=strategy['energy_threshold']
        )
        results.append({
            'strategy_name': strategy['name'],
            'output_file': output_filename,
            'total_params': total_params,
            'average_rank': average_rank,
            'merge_speed': merge_speed
        })
        print(f"策略 {strategy['name']} 统计结果:")
        print(f"  总参数量: {total_params}")
        print(f"  平均秩: {average_rank:.2f}")
        print(f"  合并速度: {merge_speed:.4f} 秒")

    print("\n--- 所有策略合并完成 ---")
    gen_report(results, "report")
    print("\n实验完成！")
