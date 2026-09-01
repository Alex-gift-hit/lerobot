import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

os.environ["https_proxy"] = "127.0.0.1:7897"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(
    model_path, torch_dtype=torch.bfloat16, _attn_implementation="eager"
).to("cuda")

"""
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Can you describe this image?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)
print(generated_texts[0])
"""

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "path": "/home/escommune/.cache/huggingface/lerobot/JiaMinEsc/env-setup_20260718_184359/videos/observation.images.left_top/chunk-000/file-000.mp4",
            },
            {"type": "text", "text": "Describe this video in detail"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=torch.bfloat16)

generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=128)
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

print(generated_texts[0])
