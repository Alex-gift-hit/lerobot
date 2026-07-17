from dataclasses import dataclass

from ..config import TeleoperatorConfig
from ..es101_leader import Es101LeaderConfigBase


@TeleoperatorConfig.register_subclass("bi_es101_leader")
@dataclass
class BiEs101LeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi SO Leader teleoperators."""

    left_arm_config: Es101LeaderConfigBase
    right_arm_config: Es101LeaderConfigBase
