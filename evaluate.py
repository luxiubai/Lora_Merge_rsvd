import os
import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, set_peft_model_state_dict
from safetensors import safe_open
from datasets import load_dataset as hf_load_dataset, Dataset, IterableDataset, IterableColumn
from tqdm import tqdm
from typing import Union, Tuple, List, Dict
from abc import ABC, abstractmethod

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

local_model_path = "model/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
lora_dir = "./lora"
local_data_dir = "data"

class DatasetLoader:
    def __init__(self, base_local_path):
        self.base_local_path = base_local_path
        self.web_dataset_map = {
            "siqa": "1-800-LLMs/siqa"
        }

    def load(self, dataset_name: str, load_method: str = "local") -> Union[Dataset, IterableDataset, IterableColumn, list, None]:
        load_method = load_method.lower()
        if load_method == "local":
            return self._load_local(dataset_name)
        elif load_method == "web":
            return self._load_web(dataset_name)
        else:
            raise ValueError(f"不支持的加载方式: {load_method}")

    def _load_local(self, dataset_name: str):
        dataset_name = dataset_name.lower()
        dataset_path = os.path.join(self.base_local_path, dataset_name)
        
        data_file = os.path.join(dataset_path, "dev.jsonl")
        labels_file = os.path.join(dataset_path, "dev-labels.lst")

        if not os.path.exists(data_file) or not os.path.exists(labels_file):
            print(f"警告: 本地数据文件 '{data_file}' 或 '{labels_file}' 未找到。")
            return None

        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            with open(labels_file, 'r', encoding='utf-8') as f:
                labels = [int(line.strip()) for line in f]
            
            if len(data) != len(labels):
                print(f"警告: 在 {dataset_name} 中，数据点数量 ({len(data)}) 与标签数量 ({len(labels)}) 不匹配。")
                return None

            for i, example in enumerate(data):
                example["label"] = labels[i]
            
            return Dataset.from_list(data)
        except Exception as e:
            print(f"加载本地数据集 '{dataset_name}' 失败: {e}")
            return None

    def _load_web(self, dataset_name: str):
        dataset_name = dataset_name.lower()
        hf_dataset_name = self.web_dataset_map.get(dataset_name, dataset_name)
        
        try:
            print(f"正在从 Hugging Face Hub 加载数据集: {hf_dataset_name}")
            dataset_dict = hf_load_dataset(hf_dataset_name, streaming=False)
            
            for split in ["validation", "test", "train"]:
                if split in dataset_dict:
                    return dataset_dict[split]
            
            print(f"警告: 在 {hf_dataset_name} 数据集中未找到 'validation', 'test', or 'train' 分割。")
            return None
        except Exception as e:
            print(f"从网络加载数据集 '{hf_dataset_name}' 失败: {e}")
            return None

class DatasetProcessor(ABC):
    @abstractmethod
    def process(self, example: Dict) -> Tuple[List[str], int, List[str]]:
        """
        Processes a single example from a dataset.

        Args:
            example (Dict): A dictionary representing a single data point.

        Returns:
            Tuple[List[str], int, List[str]]: A tuple containing:
                - choices (List[str]): The list of possible answers.
                - actual_label (int): The index of the correct answer.
                - formatted_texts (List[str]): The list of texts to be fed to the model, one for each choice.
        """
        pass

class PIQAProcessor(DatasetProcessor):
    def process(self, example: Dict) -> Tuple[List[str], int, List[str]]:
        question = example["goal"]
        choices = [example["sol1"], example["sol2"]]
        actual_label = example["label"]
        text_template = "Question: {question}\nSolution: {choice}"
        
        formatted_texts = [text_template.format(question=question, choice=c) for c in choices]
        return choices, actual_label, formatted_texts

class SIQAProcessor(DatasetProcessor):
    def process(self, example: Dict) -> Tuple[List[str], int, List[str]]:
        context = example["context"]
        question = example["question"]
        choices = [example["answerA"], example["answerB"], example["answerC"]]
        actual_label = int(example["label"]) - 1
        text_template = "Context: {context}\nQuestion: {question}\nAnswer: {choice}"

        formatted_texts = [text_template.format(context=context, question=question, choice=c) for c in choices]
        return choices, actual_label, formatted_texts

PROCESSOR_REGISTRY = {
    obj.__name__.replace("Processor", "").lower(): obj
    for obj in globals().values()
    if isinstance(obj, type) and issubclass(obj, DatasetProcessor) and obj is not DatasetProcessor
}

def get_dataset_processor(task_name: str) -> DatasetProcessor:
    task_name = task_name.lower()
    processor_class = PROCESSOR_REGISTRY.get(task_name)
    if processor_class:
        return processor_class()
    else:
        raise ValueError(f"不支持的任务或未在注册表中找到: {task_name}")

def evaluate_model(model, tokenizer, dataset, processor: DatasetProcessor, task_name: str):
    if not isinstance(dataset, (Dataset, IterableDataset)):
        print(f"警告: {task_name} 数据集类型不正确，跳过评估。")
        return 0.0

    model.eval()
    correct_predictions = 0
    total_predictions = 0

    for raw_example in tqdm(dataset, desc=f"评估 {task_name}"):
        example = dict(raw_example)
        
        _choices, actual_label, formatted_texts = processor.process(example)
        
        log_likelihoods = []
        for text in formatted_texts:
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

    data_loader = DatasetLoader(base_local_path=local_data_dir)
    piqa_processor = get_dataset_processor("piqa")
    siqa_processor = get_dataset_processor("siqa")

    piqa_dataset = data_loader.load("piqa", "local")
    if piqa_dataset is None:
        print("PIQA 数据集加载失败，无法进行评估。")
    
    siqa_dataset = data_loader.load("siqa", "web")
    if siqa_dataset is None:
        print("SIQA 数据集加载失败，无法进行评估。")
        
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    base_params = sum(p.numel() for p in base_model.parameters())
    
    start_time = time.time()
    piqa_base = evaluate_model(base_model, tokenizer, piqa_dataset, piqa_processor, "piqa")
    siqa_base = evaluate_model(base_model, tokenizer, siqa_dataset, siqa_processor, "siqa")
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
            
            piqa_score = evaluate_model(peft_model, tokenizer, piqa_dataset, piqa_processor, "piqa") if piqa_dataset is not None else 0.0
            siqa_score = evaluate_model(peft_model, tokenizer, siqa_dataset, siqa_processor, "siqa") if siqa_dataset is not None else 0.0
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
