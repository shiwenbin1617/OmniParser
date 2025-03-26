# 使用魔搭(ModelScope)下载OmniParser-v2.0模型
from modelscope import snapshot_download
import os
import shutil

# 创建目标目录
os.makedirs("weights/icon_detect", exist_ok=True)
os.makedirs("weights/icon_caption_florence", exist_ok=True)

# 从魔搭下载模型
model_dir = snapshot_download("microsoft/OmniParser-v2.0")

# 复制icon_detect文件
for file in ["train_args.yaml", "model.pt", "model.yaml"]:
    src_path = os.path.join(model_dir, "icon_detect", file)
    dst_path = os.path.join("weights/icon_detect", file)
    shutil.copy(src_path, dst_path)
    print(f"已复制: {dst_path}")

# 复制icon_caption文件
for file in ["config.json", "generation_config.json", "model.safetensors"]:
    src_path = os.path.join(model_dir, "icon_caption", file)
    dst_path = os.path.join("weights/icon_caption_florence", file)
    shutil.copy(src_path, dst_path)
    print(f"已复制: {dst_path}")

print("所有文件已下载并复制到指定位置")
