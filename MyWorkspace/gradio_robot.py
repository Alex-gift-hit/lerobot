import numpy as np

from lerobot.robots import bi_es101_follower, es101_follower

left_arm_config = es101_follower.Es101FollowerConfig(port="/dev/ttyACM1")
right_arm_config = es101_follower.Es101FollowerConfig(port="/dev/ttyACM0")

robot_config = bi_es101_follower.BiEs101FollowerConfig(left_arm_config, right_arm_config)
robot_config.id = "bi_es101"

robot = bi_es101_follower.BiEs101Follower(robot_config)

# robot.connect(calibrate=False)
robot.connect()

import gradio as gr

# ===================== 关节限位【角度 deg】
# 旋转关节统一限制 -50 ~ 50 度，夹爪不变
JOINT_LIMITS_DEG = {
    "left_shoulder_pan.pos": (-50.0, 50.0),
    "left_shoulder_lift.pos": (-50.0, 50.0),
    "left_elbow_flex.pos": (-50.0, 50.0),
    "left_wrist_flex.pos": (-50.0, 50.0),
    "left_wrist_roll.pos": (-50.0, 50.0),
    "left_wrist_yaw.pos": (-50.0, 50.0),
    "left_gripper.pos": (0.0, 1.0),
    "right_shoulder_pan.pos": (-50.0, 50.0),
    "right_shoulder_lift.pos": (-50.0, 50.0),
    "right_elbow_flex.pos": (-50.0, 50.0),
    "right_wrist_flex.pos": (-50.0, 50.0),
    "right_wrist_roll.pos": (-50.0, 50.0),
    "right_wrist_yaw.pos": (-50.0, 50.0),
    "right_gripper.pos": (0.0, 1.0),
}
joint_names = list(JOINT_LIMITS_DEG.keys())

# ===================== 全局动作【全部角度deg，直接发给机器人】
current_action = dict.fromkeys(joint_names, 0.0)
current_action["left_gripper.pos"] = 0.5
current_action["right_gripper.pos"] = 0.5

initial_obs = robot.get_observation()
initial_position = {k: v for k, v in initial_obs.items() if k.endswith(".pos")}


def random_jitter():
    """围绕0微小摆动（角度空间）"""
    new_act = {}
    for name in joint_names:
        low, high = JOINT_LIMITS_DEG[name]
        if "gripper" in name:
            val = np.clip(np.random.normal(0.5, scale=0.03), low, high)
        else:
            # 小幅抖动 ±3.4°，不会轻易碰到±50硬限位
            val = np.clip(np.random.normal(0.0, scale=3.4), low, high)
        new_act[name] = float(val)
    return new_act


def reset_zero():
    """全部旋转关节归零，夹爪0.5"""
    new_act = dict.fromkeys(joint_names, 0.0)
    new_act["left_gripper.pos"] = 0.5
    new_act["right_gripper.pos"] = 0.5
    return new_act


def send_action_btn():
    global current_action
    print("\n【发送动作 - 角度 deg】")
    for k, v in current_action.items():
        if "gripper" in k:
            print(f"{k:<26} {v:.4f}")
        else:
            print(f"{k:<26} {v:.2f} °")

    # ==================== 解除注释下发机器人 ====================
    robot.send_action(current_action)
    info_str = "✅ 动作已发送!\nAction字典(角度):\n" + str(
        {k: round(v, 3) for k, v in current_action.items()}
    )
    return info_str


def disconnect_robot_btn():
    robot.send_action(initial_position)
    robot.disconnect()
    info_str = "恢复原位，总线关闭"
    return info_str


def update_all_sliders(new_action):
    global current_action
    current_action = new_action
    outputs = []
    for jn in joint_names:
        outputs.append(gr.update(value=new_action[jn]))
    outputs.append(str({k: round(v, 3) for k, v in current_action.items()}))
    return outputs


def on_slider_change(idx, value):
    """拖动滑块直接更新（已是角度）"""
    global current_action
    jn = joint_names[idx]
    low, high = JOINT_LIMITS_DEG[jn]
    current_action[jn] = np.clip(value, low, high)
    return str({k: round(v, 3) for k, v in current_action.items()})


# ===================== 构建UI界面 =====================
with gr.Blocks(title="双臂机器人【角度直驱】控制器") as demo:
    gr.Markdown("# 🤖 双臂机器人图形化调试面板")
    gr.Markdown("> 旋转关节限制：±50° | 夹爪：0~1；输出直接给机器人使用")
    state_text = gr.Textbox(label="当前Action字典（角度）", lines=6)

    with gr.Row():
        left_col = gr.Column()
        right_col = gr.Column()

    slider_list = []
    left_joints = [n for n in joint_names if n.startswith("left_")]
    right_joints = [n for n in joint_names if n.startswith("right_")]

    with left_col:
        gr.Markdown("## 🟦 左臂")
        for jn in left_joints:
            low, high = JOINT_LIMITS_DEG[jn]
            init_val = current_action[jn]
            if "gripper" in jn:
                label = f"{jn} [0~1]"
                step = 0.01
            else:
                label = f"{jn} (°)"
                step = 0.1
            s = gr.Slider(minimum=low, maximum=high, value=init_val, label=label, step=step)
            slider_list.append(s)

    with right_col:
        gr.Markdown("## 🟥 右臂")
        for jn in right_joints:
            low, high = JOINT_LIMITS_DEG[jn]
            init_val = current_action[jn]
            if "gripper" in jn:
                label = f"{jn} [0~1]"
                step = 0.01
            else:
                label = f"{jn} (°)"
                step = 0.1
            s = gr.Slider(minimum=low, maximum=high, value=init_val, label=label, step=step)
            slider_list.append(s)

    with gr.Row():
        btn_send = gr.Button("▶️ 发送动作", variant="primary")
        btn_rand = gr.Button("🎲 随机微小抖动")
        btn_zero = gr.Button("🔄 全部归零")
        btn_disconnect = gr.Button("关机(恢复原位，关闭总线)")

    # 绑定事件
    btn_send.click(fn=send_action_btn, outputs=[state_text])
    btn_disconnect.click(fn=disconnect_robot_btn, outputs=[state_text])

    def callback_random():
        return update_all_sliders(random_jitter())

    btn_rand.click(fn=callback_random, outputs=[*slider_list, state_text])

    def callback_reset():
        return update_all_sliders(reset_zero())

    btn_zero.click(fn=callback_reset, outputs=[*slider_list, state_text])

    for idx, slider in enumerate(slider_list):
        slider.change(fn=on_slider_change, inputs=[gr.State(idx), slider], outputs=[state_text])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
