from pathlib import Path
from copy import copy
import torch
from typing import Any, TypedDict
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.configs import PreTrainedConfig

from lerobot.robots import bi_es101_follower
from lerobot.robots import es101_follower

from lerobot.datasets import (
    LeRobotDataset,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)

from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.processor import (
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
    rename_stats,
)
from lerobot.policies import smolvla

from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.feature_utils import combine_feature_dicts, hw_to_dataset_features

import os
os.environ['https_proxy'] = '127.0.0.1:7897'

def create_fake_obs() -> dict[str, Any]:
    """
    生成虚假机器人观测数据
    包含7个电机关节位置 + 3路 640×480 RGB摄像头图像数组
    Returns:
        dict: 观测字典
    """
    # 14个关节电机位置，范围模拟常见关节角度 [-np.pi, np.pi]
    joint_positions = {
        "left_shoulder_pan.pos": np.random.uniform(-np.pi, np.pi),
        "left_shoulder_lift.pos": np.random.uniform(-np.pi / 2, np.pi / 2),
        "left_elbow_flex.pos": np.random.uniform(-np.pi / 2, np.pi / 2),
        "left_wrist_flex.pos": np.random.uniform(-np.pi / 2, np.pi / 2),
        "left_wrist_roll.pos": np.random.uniform(-np.pi, np.pi),
        "left_wrist_yaw.pos": np.random.uniform(-np.pi, np.pi),
        "left_gripper.pos": np.random.uniform(0.0, 1.0),

        "right_shoulder_pan.pos": np.random.uniform(-np.pi, np.pi),
        "right_shoulder_lift.pos": np.random.uniform(-np.pi / 2, np.pi / 2),
        "right_elbow_flex.pos": np.random.uniform(-np.pi / 2, np.pi / 2),
        "right_wrist_flex.pos": np.random.uniform(-np.pi / 2, np.pi / 2),
        "right_wrist_roll.pos": np.random.uniform(-np.pi, np.pi),
        "right_wrist_yaw.pos": np.random.uniform(-np.pi, np.pi),
        "right_gripper.pos": np.random.uniform(0.0, 1.0),
    }

    #opencv 640*480 == w*h
    img_w, img_h = 640, 480
    # 生成三张随机RGB图像 shape=(H, W, 3) uint8
    image_data = {
        "left_top": np.random.randint(
            0, 256, size=(img_w, img_h, 3), dtype=np.uint8
        ),
        "left_wrist": np.random.randint(
            0, 256, size=(img_w, img_h, 3), dtype=np.uint8
        ),
        "right_wrist": np.random.randint(
            0, 256, size=(img_w, img_h, 3), dtype=np.uint8
        ),
    }
    action = joint_positions
    return {**joint_positions, **image_data}, action

def create_fake_observation_features(obs: dict[str, Any]):
    schema = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            # 图像：存入shape元组 (W,H,3)
            schema[k] = tuple(v.shape)
        else:
            # 关节浮点数：存入类型对象 float
            schema[k] = float
    return schema

# Applies a pipeline to the raw robot observation, default is IdentityProcessor



left_arm_config = es101_follower.Es101FollowerConfig(port="/dev/ttyACM1")
right_arm_config = es101_follower.Es101FollowerConfig(port="/dev/ttyACM0")

robot_config = bi_es101_follower.BiEs101FollowerConfig(left_arm_config, right_arm_config)
robot_config.id = "bi_es101"

robot = bi_es101_follower.BiEs101Follower(robot_config)

robot.connect()



_t, _r, _o = make_default_processors()
teleop_action_processor = _t
robot_action_processor =  _r
robot_observation_processor = _o

pretrained_path = Path("/home/escommune/Downloads/train/outputs/train/my_smolvla_batch64/checkpoints/010000/pretrained_model")
smolvla_config = PreTrainedConfig.from_pretrained(pretrained_path)
#smolvla_config = smolvla.configuration_smolvla.SmolVLAConfig(pretrained_path)
smolvla_policy = smolvla.modeling_smolvla.SmolVLAPolicy.from_pretrained(pretrained_name_or_path=pretrained_path, config=smolvla_config)


# Get robot observation
# obs = robot.get_observation()
obs, action = create_fake_obs()
observation_features=create_fake_observation_features(obs)
action_features = create_fake_observation_features(action)

all_obs_features = observation_features
# ``observation_features`` values are either a tuple (camera shape) or the
# ``float`` type itself used as a sentinel for scalar motor features —
# see ``dict[str, type | tuple]`` annotation on ``Robot.observation_features``.
observation_features_hw = {
    k: v
    for k, v in all_obs_features.items()
    if isinstance(v, tuple) or (v is float and k.endswith(".pos"))
}
action_features_hw = {k: v for k, v in action_features.items() if k.endswith(".pos")}

# The action side is always needed: sync inference reads action names from
# ``dataset_features[ACTION]`` to map policy tensors back to robot actions.
action_dataset_features = aggregate_pipeline_dataset_features(
    pipeline=teleop_action_processor,
    initial_features=create_initial_features(action=action_features_hw),
    use_videos=False,
)
# Observation-side aggregation is needed because of build_dataset_frame
observation_dataset_features = aggregate_pipeline_dataset_features(
    pipeline=robot_observation_processor,
    initial_features=create_initial_features(observation=observation_features_hw),
    use_videos=False
)
dataset_features = combine_feature_dicts(action_dataset_features, observation_dataset_features)

rename_map = {'observation.images.left_top': 'observation.images.camera1',
              'observation.images.left_wrist': 'observation.images.camera2',
              'observation.images.right_wrist': 'observation.images.camera3'}

preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=smolvla_config,
        pretrained_path=str(pretrained_path),
        dataset_stats=None,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )

obs_raw, action = create_fake_obs()
obs_processed = robot_observation_processor(obs_raw)

obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix="observation")
observation = copy(obs_frame)

with torch.inference_mode():
    observation = prepare_observation_for_inference(
        observation, "cuda", "transfer cube"
    )

    observation = preprocessor(observation)
    action = smolvla_policy.select_action(observation)
    action = postprocessor(action)

action_tensor = action.squeeze(0).cpu()

# Reorder to match dataset action ordering so the caller can treat
# the returned tensor uniformly across backends.
action_dict = make_robot_action(action_tensor, dataset_features)

action = {
    "left_shoulder_pan.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
    "left_shoulder_lift.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi/2, np.pi/2),
    "left_elbow_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi/2, np.pi/2),
    "left_wrist_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi/2, np.pi/2),
    "left_wrist_roll.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
    "left_wrist_yaw.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
    "left_gripper.pos": np.clip(np.random.normal(loc=0.5, scale=0.03), 0.0, 1.0),

    "right_shoulder_pan.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
    "right_shoulder_lift.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi/2, np.pi/2),
    "right_elbow_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi/2, np.pi/2),
    "right_wrist_flex.pos": np.clip(np.random.normal(loc=0, scale=0.06), -np.pi/2, np.pi/2),
    "right_wrist_roll.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
    "right_wrist_yaw.pos": np.clip(np.random.normal(loc=0, scale=0.08), -np.pi, np.pi),
    "right_gripper.pos": np.clip(np.random.normal(loc=0.5, scale=0.03), 0.0, 1.0),
}

robot.send_action(action)




