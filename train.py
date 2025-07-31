import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import time
import gc
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset as hf_load_dataset, Dataset
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_dir = r"D:\1024\M\models--Qwen--Qwen3-0.6B\snapshots\6130ef31402718485ca4d80a6234f70d9a4cf362"


piqa_data_dir = "data/piqa"
os.makedirs(piqa_data_dir, exist_ok=True)

lora_dir = "./lora"
os.makedirs(lora_dir, exist_ok=True)

def load_dataset(dataset_name) -> Dataset:
    if dataset_name.lower() == "piqa":
        print(f"从本地加载PIQA数据集: {piqa_data_dir}")
        train_file = os.path.join(piqa_data_dir, "train.jsonl")
        labels_file = os.path.join(piqa_data_dir, "train-labels.lst")
        
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = [json.loads(line) for line in f]
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = [int(line.strip()) for line in f]
        
        for i, example in enumerate(train_data):
            example["label"] = labels[i]
        
        return Dataset.from_list(train_data)
    
    elif dataset_name.lower() == "siqa":
        print("从Hugging Face镜像加载SIQA数据集: 1-800-LLMs/siqa")
        dataset = hf_load_dataset("1-800-LLMs/siqa", streaming=False)
        return dataset["train"]
    
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

tokenizer = AutoTokenizer.from_pretrained(
    model_dir,
    padding_side="right",
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_lora_config():
    return LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

def tokenize_piqa(batch):
    texts = []
    for i in range(len(batch["goal"])):
        goal = batch["goal"][i]
        sol1 = batch["sol1"][i]
        sol2 = batch["sol2"][i]
        label = batch["label"][i]
        
        correct_sol = sol1 if label == 0 else sol2
        texts.append(f"Question: {goal}\nSolution: {correct_sol}")
    return process_texts(texts)

def tokenize_siqa(batch):
    texts = []
    for i in range(len(batch["context"])):
        ctx = batch["context"][i]
        q = batch["question"][i]
        a = batch["answerA"][i]
        b = batch["answerB"][i]
        c = batch["answerC"][i]
        label = int(batch["label"][i]) - 1
        
        correct_answer = [a, b, c][label]
        
        texts.append(f"Context: {ctx}\nQuestion: {q}\nAnswer: {correct_answer}")
    return process_texts(texts)

def process_texts(texts):
    tokenized = tokenizer(
        texts,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    
    tokenized["labels"] = tokenized["input_ids"].clone()
    tokenized["labels"][tokenized["attention_mask"] == 0] = -100
    
    return tokenized

def train_lora(dataset_name):
    print("\n" + "="*50)
    print(f"=== 开始训练 {dataset_name.upper()} LoRA ===")
    print("="*50)
    
    training_args = TrainingArguments(
        output_dir=os.path.join(lora_dir, "training_results"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
        optim="adamw_torch_fused",
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        label_names=["labels"],
    )

    print(f"[{dataset_name.upper()}] 加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        use_cache=False
    ).to("cuda")

    model.gradient_checkpointing_enable()
    
    print(f"[{dataset_name.upper()}] 应用LoRA适配器...")
    peft_config = get_lora_config()
    peft_model = get_peft_model(model, peft_config)
    

    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"可训练参数: {trainable_params:,}/{total_params:,} ({100*trainable_params/total_params:.2f}%)")

    print("\n模型键名前5个示例:")
    for key in list(peft_model.state_dict().keys())[:5]:
        print(f"  - {key}")
    
    print(f"[{dataset_name.upper()}] 加载数据集...")
    dataset: Dataset = load_dataset(dataset_name)
    
    print(f"[{dataset_name.upper()}] 处理数据集...")
    torch.cuda.empty_cache()
    gc.collect()
    
    tokenize_fn = tokenize_piqa if dataset_name.lower() == "piqa" else tokenize_siqa
    
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        batch_size=500,
        remove_columns=dataset.column_names,
    )
    
    print(f"处理完成，样本数: {len(tokenized_dataset)}")
    
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )
    
    print(f"[{dataset_name.upper()}] 开始训练...")
    start_time = time.time()
    
    with tqdm(total=len(tokenized_dataset), desc=f"训练 {dataset_name.upper()}", unit="样本") as pbar:
        trainer.train()
        pbar.update(len(tokenized_dataset))
    
    duration = time.time() - start_time
    print(f"[{dataset_name.upper()}] 训练完成，耗时: {duration:.2f}秒")
    
    lora_save_dir = os.path.join(lora_dir, dataset_name.upper())
    print(f"保存LoRA适配器到: {lora_save_dir}")
    peft_model.save_pretrained(lora_save_dir)
    
    print(f"[{dataset_name.upper()}] 释放资源...")
    del peft_model, trainer
    torch.cuda.empty_cache()
    gc.collect()
    
    return duration

def main():
    datasets = ["piqa","siqa"]
    durations = {}
    
    for dataset in datasets:
        durations[dataset] = train_lora(dataset)
    
    print("\n所有训练完成!")
    print("="*30)
    print("训练用时统计:")
    for dataset, duration in durations.items():
        print(f"- {dataset.upper():<5}: {duration:.2f}秒")

if __name__ == "__main__":
    main()