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

class ReportGenerator:
    def __init__(self, tasks: List[str]):
        self.tasks = tasks
        self.header = ["模型类型"] + [f"{task.upper()} 准确率" for task in tasks] + ["评估时间", "可训练参数"]

    def get_header(self) -> str:
        return f"| {' | '.join(self.header)} |\n|{'---|' * len(self.header)}\n"

    def format_row(self, model_name: str, scores: Dict[str, float], eval_time: float, trainable_params: Union[int, None]) -> str:
        score_values = [f"{scores.get(task, 0.0):.2f}%" for task in self.tasks]
        time_str = f"{eval_time:.1f}s"
        params_str = f"{trainable_params:,}" if trainable_params is not None else "/"
        
        if not scores:
            score_values = ["适配器缺失"] * len(self.tasks)
            time_str = "/"
            params_str = "/"
        elif "error" in scores:
            score_values = ["错误"] * len(self.tasks)
            time_str = "/"
            params_str = "/"

        return f"| {model_name} | {' | '.join(score_values)} | {time_str} | {params_str} |\n"

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
    EVALUATION_TASKS = [
        {"name": "piqa", "load_method": "local"},
        {"name": "siqa", "load_method": "web"},
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    task_names = [task["name"] for task in EVALUATION_TASKS]
    report_generator = ReportGenerator(tasks=task_names)
    report_lines = ["\n\n# LoRA 评估报告\n\n", report_generator.get_header()]

    data_loader = DatasetLoader(base_local_path=local_data_dir)
    
    datasets = {}
    processors = {}
    for task in EVALUATION_TASKS:
        task_name = task["name"]
        datasets[task_name] = data_loader.load(task_name, task["load_method"])
        processors[task_name] = get_dataset_processor(task_name)
        if datasets[task_name] is None:
            print(f"{task_name.upper()} 数据集加载失败，将跳过此评估。")

    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    base_params = sum(p.numel() for p in base_model.parameters())
    
    base_scores = {}
    start_time = time.time()
    for task_name in task_names:
        if datasets[task_name]:
            base_scores[task_name] = evaluate_model(base_model, tokenizer, datasets[task_name], processors[task_name], task_name)
    base_time = time.time() - start_time
    report_lines.append(report_generator.format_row("基础模型", base_scores, base_time, base_params))
    
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
            report_lines.append(report_generator.format_row(name, {}, 0, None))
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
                report_lines.append(report_generator.format_row(name, {"error": 1}, 0, None))
                continue

            if not ranks:
                print(f"未在 {safetensors_path} 中找到LoRA层，跳过此适配器。")
                report_lines.append(report_generator.format_row(name, {}, 0, None))
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
            
            trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            
            adapter_scores = {}
            for task_name in task_names:
                if datasets[task_name]:
                    adapter_scores[task_name] = evaluate_model(peft_model, tokenizer, datasets[task_name], processors[task_name], task_name)
            
            eval_time = time.time() - start_time
            report_lines.append(report_generator.format_row(name, adapter_scores, eval_time, trainable_params))
            
        except Exception as e:
            import traceback
            print(f"评估失败: {str(e)}")
            traceback.print_exc()
            report_lines.append(report_generator.format_row(name, {"error": 1}, 0, None))
        
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
