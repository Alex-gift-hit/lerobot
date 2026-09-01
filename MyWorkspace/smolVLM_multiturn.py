import os
import sys
import time

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

os.environ["https_proxy"] = "127.0.0.1:7897"

# ============ 配置 ============
MODEL_PATH = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 64
IMAGE_SIZE = (384, 384)
# 最大保留对话轮数（每轮包含 user + assistant），设为 -1 表示不限制
MAX_HISTORY_ROUNDS = 3
# ============================


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, _attn_implementation="sdpa"
    ).to(DEVICE)
    model.eval()
    return processor, model


def preprocess_frame(frame, size=IMAGE_SIZE):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb).resize(size)
    return pil_img


def draw_text_with_wrap(
    img,
    text,
    x,
    y,
    max_width,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.7,
    font_thickness=2,
    color=(255, 255, 255),
    line_spacing=10,
):
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        test_line = current_line + (" " if current_line else "") + word
        (w, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)
        if w <= max_width:
            current_line = test_line
        else:
            if current_line:
                lines.append(current_line)
                current_line = word
            else:
                lines.append(word)
                current_line = ""
    if current_line:
        lines.append(current_line)
    (_, line_height), _ = cv2.getTextSize("Tg", font, font_scale, font_thickness)
    y_offset = 0
    for line in lines:
        cv2.putText(img, line, (x, y + y_offset), font, font_scale, color, font_thickness, cv2.LINE_AA)
        y_offset += line_height + line_spacing
    return y + y_offset


# ========== 新增：对话管理类 ==========
class ChatSession:
    def __init__(self, max_rounds=None):
        """
        :param max_rounds: 最大保留的对话轮数（user+assistant 算一轮），
                           超出时删除最早的一轮（包括图像和文本）。
        """
        self.messages = []
        self.max_rounds = max_rounds

    def add_user_message(self, image=None, text=None):
        """添加用户消息（包含图像和文本）"""
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": text})
        self.messages.append({"role": "user", "content": content})
        # 修剪历史（按轮数）
        if self.max_rounds is not None and self.max_rounds > 0:
            # 每轮由 user + assistant 组成，但可能最后一轮没有 assistant（刚添加）
            # 我们计算当前存在的完整轮数（user 数量）
            user_indices = [i for i, m in enumerate(self.messages) if m["role"] == "user"]
            if len(user_indices) > self.max_rounds:
                # 删除最早的一整轮：即从开头到第一个 user 对应的 assistant（如果有）
                first_user_idx = user_indices[0]
                # 如果第一个 user 后面紧跟 assistant，一起删；否则只删 user
                end_idx = first_user_idx + 1
                if end_idx < len(self.messages) and self.messages[end_idx]["role"] == "assistant":
                    end_idx += 1
                del self.messages[:end_idx]

    def add_assistant_message(self, text):
        # ✅ 修复：content 必须是列表
        self.messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})

    def get_response(self, processor, model, max_new_tokens=MAX_NEW_TOKENS):
        """根据当前历史生成回复，并自动保存到历史中"""
        # 准备输入
        inputs = processor.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(DEVICE, dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

        # 解码新增部分（去掉输入 prompt）
        input_len = inputs["input_ids"].shape[1]
        new_tokens = generated_ids[0][input_len:]
        response = processor.decode(new_tokens, skip_special_tokens=True)

        # 保存助手回复
        self.add_assistant_message(response)
        return response


# =====================================


def main():
    # 获取初始 prompt（可以从命令行参数获取，或在运行时修改）
    if len(sys.argv) > 1:
        current_prompt = " ".join(sys.argv[1:])
    else:
        current_prompt = input("请输入初始 prompt（例如：描述图片中的内容）：").strip()
        if not current_prompt:
            current_prompt = "Describe the image in detail."

    print(f"当前 prompt: {current_prompt}")
    print("按 'p' 修改 prompt，按 '空格' 手动推理，按 'q' 退出")
    print("连续推理模式：每 2 秒自动推理一次")

    processor, model = load_model()
    cap = cv2.VideoCapture(4)  # 根据你的摄像头调整索引
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 创建对话会话（设定最大历史轮数）
    session = ChatSession(max_rounds=MAX_HISTORY_ROUNDS)

    cv2.namedWindow("VLM Real-time Inference", cv2.WINDOW_NORMAL)

    last_inference_time = 0
    inference_interval = 600  # 自动推理间隔（秒）
    last_response = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = cv2.flip(frame, 1)

        # 显示当前 prompt 和最后回答
        cv2.putText(
            display_frame,
            f"Prompt: {current_prompt}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        if last_response:
            max_text_width = int(display_frame.shape[1] * 0.9)
            draw_text_with_wrap(
                display_frame,
                last_response,
                x=10,
                y=60,
                max_width=max_text_width,
                font_scale=0.6,
                color=(0, 255, 0),
                line_spacing=5,
            )

        cv2.imshow("VLM Real-time Inference", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # 修改 prompt（按 'p' 键）
        if key == ord("p"):
            new_prompt = input("请输入新的 prompt: ").strip()
            if new_prompt:
                current_prompt = new_prompt
                print(f"Prompt 已更新为: {current_prompt}")

        # 手动触发推理（空格）
        if key == ord(" "):
            pil_img = preprocess_frame(frame)
            print("推理中...")
            # 添加当前用户消息（图像 + prompt）
            new_prompt = input("请输入新的 prompt: ").strip()
            if new_prompt:
                current_prompt = new_prompt
                print(f"Prompt 已更新为: {current_prompt}")
            session.add_user_message(image=pil_img, text=current_prompt)
            # 生成回复
            response = session.get_response(processor, model, MAX_NEW_TOKENS)
            print(f"模型回答: {response}")
            last_response = response
            # 注意：此时历史中已包含本轮 user 和 assistant，下次推理会继承
            continue

        # 自动推理（按时间间隔）
        current_time = time.time()
        if inference_interval > 0 and (current_time - last_inference_time) > inference_interval:
            pil_img = preprocess_frame(frame)
            print("自动推理...")
            session.add_user_message(pil_img, current_prompt)
            response = session.get_response(processor, model, MAX_NEW_TOKENS)
            print(f"模型回答: {response}")
            last_response = response
            last_inference_time = current_time

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
