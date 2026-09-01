import os

import pandas as pd
from transformers import AutoTokenizer

os.environ["https_proxy"] = "127.0.0.1:7897"
# ========== 配置 ==========
CSV_PATH = "inputs__input_ids___cpu_.csv"

# 图像特殊token映射（你提供的字典）
SPECIAL_IMG_TOKENS = {
    49279: "<end_of_utterance>",
    49189: "<fake_token_around_image>",
    49152: "<global-img>",
    49153: "<row_1_col_1>",
    49154: "<row_1_col_2>",
    49155: "<row_1_col_3>",
    49156: "<row_1_col_4>",
    49159: "<row_2_col_1>",
    49160: "<row_2_col_2>",
    49161: "<row_2_col_3>",
    49162: "<row_2_col_4>",
    49165: "<row_3_col_1>",
    49166: "<row_3_col_2>",
    49167: "<row_3_col_3>",
    49168: "<row_3_col_4>",
    49171: "<row_4_col_1>",
    49172: "<row_4_col_2>",
    49173: "<row_4_col_3>",
    49174: "<row_4_col_4>",
    49190: "<image>",
}

# 1. 加载tokenizer

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Instruct")

# 2. 读取CSV input_ids
df = pd.read_csv(CSV_PATH, header=None)
# 展平成一维列表
input_ids = df.values.flatten().tolist()

# 3. 分段解码：区分普通文本token & 图像特殊token
decoded_parts = []
for tid in input_ids:
    if tid in SPECIAL_IMG_TOKENS:
        # 图像专用特殊token，直接替换成可读标记
        decoded_parts.append(SPECIAL_IMG_TOKENS[tid])
    else:
        # 普通文本token，调用tokenizer单id解码
        text_tok = tokenizer.decode([tid], skip_special_tokens=False)
        decoded_parts.append(text_tok)

# 4. 拼接完整可读字符串
full_text = " ".join(decoded_parts)

# 5. 输出结果
print("===== 完整解码序列（可读版）=====")
print(full_text)

# 可选：保存到文本文件方便查看
with open("smolvlm_decoded.txt", "w", encoding="utf-8") as f:
    f.write(full_text)

# 可选：统计各类token数量
from collections import Counter

cnt = Counter(input_ids)
print("\n===== Token 统计 Top20 =====")
for tok_id, num in cnt.most_common(20):
    desc = SPECIAL_IMG_TOKENS.get(tok_id, tokenizer.decode([tok_id]))
    print(f"{tok_id:6d} | {num:4d} 次 | {desc}")
