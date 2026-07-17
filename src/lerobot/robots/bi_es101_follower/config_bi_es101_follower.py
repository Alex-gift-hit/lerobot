from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig
from ..es101_follower import Es101FollowerConfigBase


@RobotConfig.register_subclass("bi_es101_follower")
@dataclass
class BiEs101FollowerConfig(RobotConfig):
    """Configuration class for Bi Es101 Follower robots."""

    # use configbase to avoid recursive parse(RobotConfig → 可选 bi_es101_follower 子类 → 包含 left_arm_config（RobotConfig 子类）→ 又可选 bi_es101_follower → 再次嵌套手臂字段 → ……)
    left_arm_config: Es101FollowerConfigBase
    right_arm_config: Es101FollowerConfigBase

    # Top-level cameras not attached to a specific side. Keys are kept as-is in
    # observations (no `left_`/`right_` prefix). Per-arm cameras (declared on
    # `{left,right}_arm_config.cameras`) are prefixed.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
