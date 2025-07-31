import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm
import os
import time
import math
from typing import Tuple, Dict, Any, List

def _move_tensors_to_device(state_dict: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            state_dict[key] = value.to(device)
    return state_dict

def _get_lora_module_names(state_dict: Dict[str, torch.Tensor], up_weight_suffix: str) -> set:
    module_names = set()
    for key in state_dict.keys():
        if key.endswith(up_weight_suffix):
            module_names.add(key.replace(up_weight_suffix, ''))
    return module_names

def get_lora_rank(lora_down_weight: torch.Tensor) -> int:
    return lora_down_weight.shape[0]

def count_lora_parameters(lora_up_weight: torch.Tensor, lora_down_weight: torch.Tensor) -> int:
    return lora_up_weight.numel() + lora_down_weight.numel()

def concatenate_merge(
    lora_up1: torch.Tensor, lora_down1: torch.Tensor, alpha1: float, r1: int,
    lora_up2: torch.Tensor, lora_down2: torch.Tensor, alpha2: float, r2: int,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    lora_up_new = torch.cat([lora_up1, lora_up2], dim=1) 
    lora_down_new = torch.cat([lora_down1, lora_down2], dim=0) 
    new_rank = r1 + r2
    return lora_up_new, lora_down_new, new_rank

def geometric_mean_merge(
    lora_up1: torch.Tensor, lora_down1: torch.Tensor, alpha1: float, r1: int,
    lora_up2: torch.Tensor, lora_down2: torch.Tensor, alpha2: float, r2: int,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    epsilon = 1e-8
    with torch.no_grad():
        sign_up = torch.sign(lora_up1 * lora_up2)
        merged_up = sign_up * torch.sqrt(torch.abs(lora_up1) * torch.abs(lora_up2) + epsilon)
        
        sign_down = torch.sign(lora_down1 * torch.abs(lora_down2))
        merged_down = sign_down * torch.sqrt(torch.abs(lora_down1) * torch.abs(lora_down2) + epsilon)
        
        new_rank = min(r1, r2)
        return merged_up[:, :new_rank], merged_down[:new_rank, :], new_rank

def svd_merge(
    lora_up1: torch.Tensor, lora_down1: torch.Tensor, alpha1: float, r1: int,
    lora_up2: torch.Tensor, lora_down2: torch.Tensor, alpha2: float, r2: int,
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    W1 = lora_up1 @ lora_down1 * (alpha1 / max(r1, 1))
    W2 = lora_up2 @ lora_down2 * (alpha2 / max(r2, 1))
    merged_W = W1 + W2
    
    U, S, Vt = torch.linalg.svd(merged_W, full_matrices=False)
    
    new_rank = min(r1, r2)
    
    sqrt_S = torch.sqrt(S[:new_rank])
    lora_up_new = U[:, :new_rank] @ torch.diag(sqrt_S)
    lora_down_new = torch.diag(sqrt_S) @ Vt[:new_rank, :]
    
    return lora_up_new, lora_down_new, new_rank

def weighted_merge(
    lora_up1: torch.Tensor, lora_down1: torch.Tensor, alpha1: float, r1: int,
    lora_up2: torch.Tensor, lora_down2: torch.Tensor, alpha2: float, r2: int,
    weight: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    W1 = lora_up1 @ lora_down1 * (alpha1 / max(r1, 1))
    W2 = lora_up2 @ lora_down2 * (alpha2 / max(r2, 1))
    merged_W = weight * W1 + (1 - weight) * W2
    
    new_rank = min(r1, r2)
    U, S, Vt = torch.linalg.svd(merged_W, full_matrices=False)
    
    sqrt_S = torch.sqrt(S[:new_rank])
    lora_up_new = U[:, :new_rank] @ torch.diag(sqrt_S)
    lora_down_new = torch.diag(sqrt_S) @ Vt[:new_rank, :]
    
    return lora_up_new, lora_down_new, new_rank

def merge_lora_adapters_covention(
    lora_path1: str, 
    lora_path2: str, 
    output_dir: str,
    merge_methods: Dict[str, Any], # 新增参数
    up_weight_suffix: str = ".lora_B.weight",
    down_weight_suffix: str = ".lora_A.weight",
    alpha_suffix: str = ".alpha",
    merge_weight: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"键名配置: up={up_weight_suffix}, down={down_weight_suffix}, alpha={alpha_suffix}")
    
    try:
        lora1_state_dict = load_file(lora_path1)
        lora2_state_dict = load_file(lora_path2)
    except Exception as e:
        raise RuntimeError(f"加载LoRA文件失败: {e}")
    
    lora1_state_dict = _move_tensors_to_device(lora1_state_dict, device)
    lora2_state_dict = _move_tensors_to_device(lora2_state_dict, device)

    all_module_names = _get_lora_module_names(lora1_state_dict, up_weight_suffix)
    all_module_names.update(_get_lora_module_names(lora2_state_dict, up_weight_suffix))
    print(f"发现 {len(all_module_names)} 个LoRA模块")

    merge_results = {}

    for method_name, merge_func in merge_methods.items():
        print(f"\n=== 执行 {method_name} 合并算法 ===")
        start_time = time.time()
        merged_state_dict = {}
        total_params = 0
        total_rank = 0
        module_count = 0
        skipped_modules = 0
        
        kwargs = {"weight": merge_weight} if method_name == "weighted_avg" else {}

        for module_name in tqdm(all_module_names, desc=f"处理模块 ({method_name})"):
            up_key = f"{module_name}{up_weight_suffix}"
            down_key = f"{module_name}{down_weight_suffix}"
            alpha_key = f"{module_name}{alpha_suffix}"

            has_lora1 = up_key in lora1_state_dict and down_key in lora1_state_dict
            has_lora2 = up_key in lora2_state_dict and down_key in lora2_state_dict

            if has_lora1 and has_lora2:
                up1 = lora1_state_dict[up_key].to(device, torch.float32)
                down1 = lora1_state_dict[down_key].to(device, torch.float32)
                up2 = lora2_state_dict[up_key].to(device, torch.float32)
                down2 = lora2_state_dict[down_key].to(device, torch.float32)
                
                alpha1 = lora1_state_dict.get(alpha_key, torch.tensor(1.0, device=device)).item()
                r1 = get_lora_rank(down1)
                alpha2 = lora2_state_dict.get(alpha_key, torch.tensor(1.0, device=device)).item()
                r2 = get_lora_rank(down2)
                
                try:
                    up_new, down_new, new_rank = merge_func(
                        up1, down1, alpha1, r1,
                        up2, down2, alpha2, r2,
                        **kwargs
                    )
                    
                    merged_state_dict[up_key] = up_new.to(up1.dtype).contiguous()
                    merged_state_dict[down_key] = down_new.to(up1.dtype).contiguous()
                    
                    if alpha_key in lora1_state_dict or alpha_key in lora2_state_dict:
                        merged_state_dict[alpha_key] = torch.tensor([alpha1], device=device)
                    
                    total_params += count_lora_parameters(up_new, down_new)
                    total_rank += new_rank
                    module_count += 1
                except Exception as e:
                    print(f"\n模块 {module_name} 合并失败: {e}")
                    skipped_modules += 1
                    continue
                    
            elif has_lora1:
                merged_state_dict[up_key] = lora1_state_dict[up_key]
                merged_state_dict[down_key] = lora1_state_dict[down_key]
                if alpha_key in lora1_state_dict:
                    merged_state_dict[alpha_key] = lora1_state_dict[alpha_key]
                
                params = count_lora_parameters(lora1_state_dict[up_key], lora1_state_dict[down_key])
                rank = get_lora_rank(lora1_state_dict[down_key])
                total_params += params
                total_rank += rank
                module_count += 1
                
            elif has_lora2:
                merged_state_dict[up_key] = lora2_state_dict[up_key]
                merged_state_dict[down_key] = lora2_state_dict[down_key]
                if alpha_key in lora2_state_dict:
                    merged_state_dict[alpha_key] = lora2_state_dict[alpha_key]
                
                params = count_lora_parameters(lora2_state_dict[up_key], lora2_state_dict[down_key])
                rank = get_lora_rank(lora2_state_dict[down_key])
                total_params += params
                total_rank += rank
                module_count += 1

        for key, value in lora1_state_dict.items():
            if not any(key.endswith(suffix) for suffix in [up_weight_suffix, down_weight_suffix, alpha_suffix]):
                merged_state_dict[key] = value
                
        for key, value in lora2_state_dict.items():
            if not any(key.endswith(suffix) for suffix in [up_weight_suffix, down_weight_suffix, alpha_suffix]):
                if key not in merged_state_dict:
                    merged_state_dict[key] = value

        method_output_dir = os.path.join(output_dir, method_name)
        os.makedirs(method_output_dir, exist_ok=True)
        output_filename = os.path.join(method_output_dir, "adapter_model.safetensors")
        
        try:
            save_file(merged_state_dict, output_filename)
            end_time = time.time()
            merge_time = end_time - start_time
            avg_rank = total_rank / module_count if module_count > 0 else 0
            
            merge_results[method_name] = {
                "output_file": output_filename,
                "total_parameters": total_params,
                "average_rank": avg_rank,
                "merge_time": merge_time,
                "modules_merged": module_count,
                "modules_skipped": skipped_modules
            }
            
            print(f"保存成功: {output_filename}")
            print(f"    参数总量: {total_params:,} | 平均秩: {avg_rank:.2f} | 耗时: {merge_time:.2f}s")
        except Exception as e:
            print(f"保存失败: {e}")
            merge_results[method_name] = {"error": str(e)}

    return merge_results

def generate_report(results: Dict[str, Dict[str, Any]], output_dir: str):
    report_path = os.path.join(output_dir, "lora_merge_report.md")
    
    file_exists = os.path.exists(report_path)
    file_is_empty = not file_exists or os.stat(report_path).st_size == 0

    with open(report_path, "a", encoding="utf-8") as f:
        if file_is_empty:
            f.write("# LoRA 合并实验报告\n\n")
            f.write("| 方法 | 参数总量 | 平均秩 | 合并耗时(s) |\n")
            f.write("|------|---------|-------|--------|\n")
        
        for method, data in results.items():
            if "error" in data:
                f.write(f"| {method} | - | - | - |\n")
            else:
                f.write(
                    f"| {method} | {data['total_parameters']:,} | "
                    f"{data['average_rank']:.2f} | {data['merge_time']:.2f} |\n"
                )
    
    print(f"\n实验报告已保存至: {report_path}")

if __name__ == "__main__":
    lora_path_piqa = "lora/PIQA/adapter_model.safetensors"
    lora_path_siqa = "lora/SIQA/adapter_model.safetensors"
    output_directory = "lora"
    
    if not os.path.exists(lora_path_piqa):
        raise FileNotFoundError(f"未找到LoRA文件: {lora_path_piqa}")
    if not os.path.exists(lora_path_siqa):
        raise FileNotFoundError(f"未找到LoRA文件: {lora_path_siqa}")
    
    print(f"实验配置:")
    print(f"  - LoRA 1: {lora_path_piqa}")
    print(f"  - LoRA 2: {lora_path_siqa}")
    print(f"  - 输出目录: {output_directory}\n")

    merge_methods_main = {
        "M_Concatenate": concatenate_merge,
        "M_Geometric_mean": geometric_mean_merge,
        "M_SVD": svd_merge,
        "M_Weighted_avg": weighted_merge
    }
    
    experiment_results = merge_lora_adapters_covention(
        lora_path1=lora_path_piqa,
        lora_path2=lora_path_siqa,
        output_dir=output_directory,
        merge_methods=merge_methods_main,
        up_weight_suffix=".lora_B.weight",
        down_weight_suffix=".lora_A.weight",
        alpha_suffix=".alpha",
        merge_weight=0.5 
    )
    
    generate_report(experiment_results, "report")
    print("\n实验完成！")