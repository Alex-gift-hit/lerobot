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
# 推理时的最大输出 token 数（越大回答越长，但也越慢）
MAX_NEW_TOKENS = 64
# 输入图像缩放尺寸（可根据显存调整）
IMAGE_SIZE = (384, 384)
# ============================


def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa",  # 某些环境下可改用 "flash_attention_2" 加速
    ).to(DEVICE)
    model.eval()
    return processor, model


def preprocess_frame(frame, size=IMAGE_SIZE):
    """将 OpenCV 的 BGR 帧转为 RGB 的 PIL 图像，并缩放"""
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
    """按像素宽度自动换行绘制文本（英文/ASCII）"""
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


def run_inference(processor, model, image, prompt):
    """对单张图片进行 VLM 推理"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},  # 占位符
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # 处理输入（需传入图片列表）
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(DEVICE, dtype=torch.bfloat16)

    # 生成回答
    with torch.no_grad():
        generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS)

    # 解码输出（跳过特殊 token）
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 去掉可能包含的提示部分（返回的是完整对话，可只提取助手回复）
    # 简单方法：按 "Assistant:" 分割取最后一段
    if "Assistant:" in generated_text:
        generated_text = generated_text.split("Assistant:")[-1].strip()
    return generated_text


def main():
    # 获取提示词（可从命令行参数读入，或运行时交互输入）
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = input("请输入 prompt（例如：描述图片中的内容）：").strip()
        if not prompt:
            prompt = "Describe the image in detail."  # 默认提示

    print(f"使用提示词: {prompt}")
    print("正在加载模型...")
    processor, model = load_model()

    # 打开摄像头（0 通常为默认摄像头，可改为视频文件路径）
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    # 设置窗口
    cv2.namedWindow("VLM Real-time Inference", cv2.WINDOW_NORMAL)

    print("按 'q' 退出，按 '空格' 触发一次推理（手动模式）")
    print("连续推理模式：每 10 秒自动推理一次（可按需调整）")

    # 性能控制：记录上次推理时间，避免连续卡顿
    last_inference_time = 0
    inference_interval = 10.0  # 自动推理间隔（秒），设 0 为每帧推理（会很慢）

    # 存储最近一次推理结果，用于在窗口上显示
    last_response = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("摄像头读取失败")
            break

        # 显示原始画面（镜像翻转更自然）
        display_frame = cv2.flip(frame, 1)

        # 如果有最近一次的回答，叠加到画面上（仅支持英文，如需中文见注释）
        if last_response:
            max_text_width = int(display_frame.shape[1] * 0.9)
            draw_text_with_wrap(
                display_frame,
                last_response,
                x=10,
                y=30,
                max_width=max_text_width,
                font_scale=0.6,
                color=(0, 255, 0),
                line_spacing=5,
            )

        cv2.imshow("VLM Real-time Inference", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        # 手动触发模式：按空格键进行一次推理
        if key == ord(" "):
            pil_img = preprocess_frame(frame)
            print("推理中...")
            response = run_inference(processor, model, pil_img, prompt)
            print(f"模型回答: {response}")
            last_response = response
            continue

        # 自动推理模式：按时间间隔推理
        current_time = time.time()
        if inference_interval > 0 and (current_time - last_inference_time) > inference_interval:
            pil_img = preprocess_frame(frame)
            print("自动推理...")
            response = run_inference(processor, model, pil_img, prompt)
            print(f"模型回答: {response}")
            last_response = response
            last_inference_time = current_time

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
