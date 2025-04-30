from typing import Protocol 


from fermbot.common.robot_devices.robots.configs import (
    ManipulatorRobotConfig,
    RobotConfig, 
    So100RobotConfig,
)

def get_arm_id(name, arm_type):
    """Returns the string identifier of a robot arm."""
    return f"{name}_{arm_type}"


class Robot(Protocol):
    robot_type: str
    features: dict 

    def connect(self): ...
    def run_calibration(self): ...
    def teleop_step(self, record_data=False): ...
    def capture_observation(self): ...
    def send_action(self, action): ...
    def disconnect(self): ...


def make_robot_config(robot_type: str, **kwargs) -> RobotConfig:
    if robot_type == "so100":
        return So100RobotConfig(**kwargs)
    else:
        raise ValueError(f"Robot type '{robot_type} is not available")
    

def make_robot_from_config(config: RobotConfig):
    if isinstance(config, ManipulatorRobotConfig):
        from fermbot.common.robot_devices.robots.manipulator import ManipulatorRobot
