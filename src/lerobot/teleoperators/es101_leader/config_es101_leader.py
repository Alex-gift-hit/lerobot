from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("es101_leader")
@dataclass
class Es101LeaderConfig(TeleoperatorConfig):
    """Base configuration class for ES Leader teleoperators."""

    # Port to connect to the arm
    port: str

    # Whether to use degrees for angles
    use_degrees: bool = True
