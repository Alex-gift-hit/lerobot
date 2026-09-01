import numpy as np

from lerobot.robots import bi_es101_follower, es101_follower

left_arm_config = es101_follower.Es101FollowerConfig(port="/dev/ttyACM1")
right_arm_config = es101_follower.Es101FollowerConfig(port="/dev/ttyACM0")

robot_config = bi_es101_follower.BiEs101FollowerConfig(left_arm_config, right_arm_config)
robot_config.id = "bi_es101"

robot = bi_es101_follower.BiEs101Follower(robot_config)

# robot.connect(calibrate=False)
robot.connect()


def get_action():
    action = {
        "left_shoulder_pan.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
        "left_shoulder_lift.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi / 2, np.pi / 2),
        "left_elbow_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi / 2, np.pi / 2),
        "left_wrist_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi / 2, np.pi / 2),
        "left_wrist_roll.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
        "left_wrist_yaw.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
        "left_gripper.pos": np.clip(np.random.normal(loc=0.5, scale=0.03), 0.0, 1.0),
        "right_shoulder_pan.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
        "right_shoulder_lift.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi / 2, np.pi / 2),
        "right_elbow_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi / 2, np.pi / 2),
        "right_wrist_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi / 2, np.pi / 2),
        "right_wrist_roll.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
        "right_wrist_yaw.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
        "right_gripper.pos": np.clip(np.random.normal(loc=0.5, scale=0.03), 0.0, 1.0),
    }
    return action


action = get_action()
robot.send_action(action)

while True:
    print("hello world")
    input()
    robot.send_action(action)
