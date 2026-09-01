# extract_vlm_from_pi05.py
import os

os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 关键：屏蔽 GPU
import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from lerobot.policies.pi05 import PI05Policy  # 加载 Pi0.5 的库

device = torch.device("cpu")


# ------------------------------------------------------------
# 1. 加载你本地的 Pi0.5 模型（或 Hugging Face 上的）
# ------------------------------------------------------------
pi05_path = "lerobot/pi05_base"  # 换成你本地的路径，比如 "/home/xxx/pi05_checkpoint"
print("正在加载 Pi0.5 模型...")
policy = PI05Policy.from_pretrained(pi05_path).to(device).eval()
print("Pi0.5 加载完成")

# ------------------------------------------------------------
# 2. 从 Pi0.5 中抽取出 PaliGemma 的权重
# ------------------------------------------------------------
pi05_state = policy.state_dict()

# 关键：找出 PaliGemma 权重的键名前缀（不同版本可能不同）
# 打印前几个键名看看结构，方便你确认
# print([k for k in pi05_state.keys() if 'paligemma' in k][:5])

# 通常 LeRobot 实现里前缀是 "model.paligemma."，我们把它过滤出来
paligemma_weights = {}
for key, value in pi05_state.items():
    if "paligemma" in key:  # 模糊匹配，确保不漏
        # 去掉前缀，还原成标准 PaliGemma 的键名
        new_key = key.replace("model.paligemma.", "").replace("paligemma.", "")
        paligemma_weights[new_key] = value

print(f"抽取到 {len(paligemma_weights)} 个权重层")

# ------------------------------------------------------------
# 3. 创建标准 Hugging Face 的 PaliGemma 模型，并注入权重
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("正在初始化标准 PaliGemma 模型...")
vlm_model = PaliGemmaForConditionalGeneration.from_pretrained(
    "google/paligemma-3b-pt-224",  # 用这个骨架来接权重
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# 把抽出来的权重塞进去（strict=False 防止极个别层名对不上）
vlm_model.load_state_dict(paligemma_weights, strict=False)
vlm_model.eval()
print("权重注入完成！")

# ------------------------------------------------------------
# 4. 后面的流程跟你之前的代码一模一样
# ------------------------------------------------------------
processor = AutoProcessor.from_pretrained("google/paligemma-3b-pt-224", use_fast=False)

url = "/home/escommune/Downloads/a.png"
image = Image.open(url)

prompt = "ocr"  # 想干嘛就改这里，比如 "describe the image"
model_inputs = processor(text=prompt, images=image, return_tensors="pt").to(vlm_model.device)
input_len = model_inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = vlm_model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
    generation = generation[0][input_len:]
    decoded = processor.decode(generation, skip_special_tokens=True)
    print(decoded)
