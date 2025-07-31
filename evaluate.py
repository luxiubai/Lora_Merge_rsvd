import os
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, set_peft_model_state_dict
from safetensors import safe_open
from datasets import load_dataset as hf_load_dataset, Dataset, IterableDataset, IterableColumn
from tqdm import tqdm
from typing import Union

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

local_model_path = "model/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
lora_dir = "./lora"
piqa_data_dir = "data/piqa"

def load_evaluation_dataset(dataset_name) -> Union[Dataset, IterableDataset, IterableColumn, list, None]:
    if dataset_name.lower() == "piqa":
        data_file = os.path.join(piqa_data_dir, "dev.jsonl")
        labels_file = os.path.join(piqa_data_dir, "dev-labels.lst")

        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f]
        
        for i, example in enumerate(data):
            example["label"] = labels[i]
        
        return Dataset.from_list(data)
    elif dataset_name.lower() == "siqa":
        dataset_dict = hf_load_dataset("1-800-LLMs/siqa", streaming=False)
        if "validation" in dataset_dict:
            return dataset_dict["validation"]
        else:
            print("警告: SIQA 数据集未找到 'validation' 分割。")
            return None
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

def evaluate_model(model, tokenizer, dataset, task_name):
    if not isinstance(dataset, (Dataset, IterableDataset)):
        print(f"警告: {task_name} 数据集类型不正确，跳过评估。")
        return 0.0

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    for raw_example in tqdm(dataset, desc=f"评估 {task_name}"):
        example = dict(raw_example)
        
        if task_name.lower() == "piqa":
            question = example["goal"]
            choices = [example["sol1"], example["sol2"]]
            actual_label = example["label"]
            text_template = "Question: {question}\nSolution: {choice}"
        else:
            choices = [example["answerA"], example["answerB"], example["answerC"]]
            actual_label = int(example["label"]) - 1
            text_template = "Context: {context}\nQuestion: {question}\nAnswer: {choice}"
        
        log_likelihoods = []
        for choice in choices:
            if task_name.lower() == "piqa":
                text = text_template.format(question=question, choice=choice)
            else:
                text = text_template.format(
                    context=example["context"],
                    question=example["question"],
                    choice=choice
                )
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs.input_ids)
                log_likelihood = -outputs.loss.item()
            
            log_likelihoods.append(log_likelihood)
        
        predicted_label = log_likelihoods.index(max(log_likelihoods))
        if predicted_label == actual_label:
            correct_predictions += 1
        total_predictions += 1
    
    return (correct_predictions / total_predictions) * 100

def main():
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.padding_side = "left"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report_lines = ["\n\n# LoRA 评估报告\n\n"]
    report_lines.append("| 模型类型 | PIQA 准确率 | SIQA 准确率 | 评估时间 | 可训练参数 |\n")
    report_lines.append("|---|---|---|---|---|\n")

    piqa_dataset = load_evaluation_dataset("piqa")
    if piqa_dataset is None:
        print("PIQA 数据集加载失败，无法进行评估。")
    
    siqa_dataset = load_evaluation_dataset("siqa")
    if siqa_dataset is None:
        print("SIQA 数据集加载失败，无法进行评估。")
        
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    base_params = sum(p.numel() for p in base_model.parameters())
    
    start_time = time.time()
    piqa_base = evaluate_model(base_model, tokenizer, piqa_dataset, "piqa")
    siqa_base = evaluate_model(base_model, tokenizer, siqa_dataset, "siqa")
    base_time = time.time() - start_time
    report_lines.append(f"| 基础模型 | {piqa_base:.2f}% | {siqa_base:.2f}% | {base_time:.1f}s | {base_params:,} |\n")
    
    del base_model
    torch.cuda.empty_cache()

    adapters = [
        ("PIQA LoRA", "PIQA"),
        ("SIQA LoRA", "SIQA"),
        ("M_Concatenate LoRA", "M_Concatenate"),
        ("M_Geometric_mean LoRA", "M_Geometric_mean"),
        ("M_SVD LoRA", "M_SVD"),
        ("M_Weighted_avg LoRA", "M_Weighted_avg"),
        ("M_rSVD1 LoRA", "M_rSVD_strategy1"),
        ("M_rSVD2 LoRA", "M_rSVD_strategy2"),
        ("M_rSVD3 LoRA", "M_rSVD_strategy3"),
    ]

    for name, adapter_name in adapters:
        print(f"\n=== 评估 {name} ===")
        start_time = time.time()
        adapter_path = os.path.join(lora_dir, adapter_name)
        
        if not os.path.exists(adapter_path):
            print(f"适配器目录不存在: {adapter_path}")
            report_lines.append(f"| {name} | 适配器缺失 | 适配器缺失 | / | / |\n")
            continue
        
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            safetensors_path = os.path.join(adapter_path, "adapter_model.safetensors")
            
            ranks = {}
            try:
                with safe_open(safetensors_path, framework="pt") as f:
                    lora_A_keys = [k for k in f.keys() if k.endswith(".lora_A.weight")]
                    for key in lora_A_keys:
                        module_name = key.rsplit('.', 2)[0]
                        module_name = module_name.replace("base_model.model.", "").replace("model.", "")
                        rank = f.get_tensor(key).shape[0]
                        ranks[module_name] = rank
            except Exception as e:
                print(f"无法解析safetensors文件 {safetensors_path}: {e}")
                report_lines.append(f"| {name} | safetensors解析错误 | safetensors解析错误 | / | / |\n")
                continue

            if not ranks:
                print(f"未在 {safetensors_path} 中找到LoRA层，跳过此适配器。")
                report_lines.append(f"| {name} | 无LoRA层 | 无LoRA层 | / | / |\n")
                continue

            lora_config = LoraConfig(
                r=max(ranks.values()) if ranks else 4,
                lora_alpha=16,
                target_modules=list(ranks.keys()),
                rank_pattern=ranks,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                inference_mode=True
            )

            peft_model = PeftModel(base_model, lora_config, adapter_name="default")

            adapter_weights = {}
            with safe_open(safetensors_path, framework="pt") as f:
                for key in f.keys():
                    if "lora" in key.lower():
                        adapter_weights[key] = f.get_tensor(key)
            
            set_peft_model_state_dict(peft_model, adapter_weights)
            
            peft_model.set_adapter("default")
            
            total_params = sum(p.numel() for p in peft_model.parameters())
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
            
            piqa_score = evaluate_model(peft_model, tokenizer, piqa_dataset, "piqa") if piqa_dataset is not None else 0.0
            siqa_score = evaluate_model(peft_model, tokenizer, siqa_dataset, "siqa") if siqa_dataset is not None else 0.0
            eval_time = time.time() - start_time
            
            report_lines.append(
                f"| {name} | {piqa_score:.2f}% | {siqa_score:.2f}% | {eval_time:.1f}s | {trainable_params:,} |\n"
            )
            
        except Exception as e:
            import traceback
            print(f"评估失败: {str(e)}")
            traceback.print_exc()
            report_lines.append(f"| {name} | 错误 | 错误 | / | / |\n")
        
        finally:
            if 'base_model' in locals():
                del base_model
            if 'peft_model' in locals():
                del peft_model
            torch.cuda.empty_cache()

    report_path = os.path.join("report", "lora_evaluation_report.md")
    os.makedirs("report", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(report_lines))
    print(f"\n评估完成! 报告已保存为 {report_path}")

if __name__ == "__main__":
    main()
