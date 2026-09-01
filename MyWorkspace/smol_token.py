import os

os.environ["https_proxy"] = "127.0.0.1:7897"
from transformers import AutoTokenizer

# SmolVLM2内置tokenizer = SmolLM2分词器
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")

text = "transfer cube\n"
ids = tok.encode(text, add_special_tokens=False)
tokens = tok.convert_ids_to_tokens(ids)

print("ids:", ids)
print("tokens:", tokens)
# 你会看到输出包含：'transfer', 'Ġcube', 'Ċ'
print("解码验证:", repr(tok.decode(ids)))
# 解码结果: 'transfer cube\n'

"""
32154, 20636,   198,     2,     2,     2,     2,     2,     2,     2,
2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
2,     2,     2,     2,     2,     2,     2,     2
"""
