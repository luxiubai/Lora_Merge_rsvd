import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0" 

from huggingface_hub import snapshot_download

target_dir = "./model"

os.makedirs(target_dir, exist_ok=True)

model_path = snapshot_download(
    repo_id="Qwen/Qwen3-0.6B",
    cache_dir=target_dir,
    local_dir_use_symlinks=True,
    max_workers=2,
    token=None
)

print(f"模型已下载至：{model_path}")
