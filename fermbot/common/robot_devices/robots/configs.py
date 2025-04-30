import abc 
from dataclasses import dataclass, field 
from typing import Sequence 

import draccus 

from fermbot.common.robot_devices.cameras.configs import (
    CameraConfig, 
    IntelRealSenseCameraConfig, 
    OpenCVCameraConfig,
)

from fermbot.common.robot_devices.motors.configs import (
    FeetechMotorsBusConfig,
    MotorsBusConfig,
)
import fermia_camera.realsense_camera as realsense_camera
import fermia_camera.usb_camera as usb_camera

@dataclass 
class RobotConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    


@dataclass 
class ManipulatorRobotConfig(RobotConfig):
    leader_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    follower_arms: dict[str, MotorsBusConfig] = field(default_factory=lambda: {})
    
    max_relative_target: list[float] | float | None = None 

    gripper_open_degree: float | None = None 

    mock: bool = False 

    def __post_init__(self):
        if self.mock:
            for arm in self.leader_arms.values():
                if not arm.mock:
                    arm.mock = True 
            for arm in self.follower_arms.values():
                if not arm.mock:
                    arm.mock = True 

        if self.max_relative_target is not None and isinstance(self.max_relative_target, Sequence):
            for name in self.follower_arms:
                if len(self.follower_arms[name].motors) != len(self.max_relative_target):
                    raise ValueError(
                        f"len(max_relative_target)={len(self.max_relative_target)} but the follower arm with name {name} has "
                        f"{len(self.follower_arms[name].motors)} motors. Please make sure that the "
                        f"`max_relative_target` list has as many parameters as there are motors per arm. "
                        "Note: This feature does not yet work with robots where different follower arms have "
                        "different numbers of motors."
                    )
                


@RobotConfig.register_subclass("so100")
@dataclass
class So100RobotConfig(ManipulatorRobotConfig):
    calibration_dir: str = ".cache/calibration/so100"

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes 
    # `relative positional target vector` in the context of attention mechanisms in Transforrmer models, represents the relative distance between tokens in a sequence
    # 
    #  Set this to a positive scalar to have the same values for all motors, or a list that is the same length as the number of motors in your follower arms 
    
    max_relative_target: int | None = None
    
    leader_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM1",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    follower_arms: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": FeetechMotorsBusConfig(
                port="/dev/ttyACM0",
                motors={
                    # name: (index, model)
                    "shoulder_pan": [1, "sts3215"],
                    "shoulder_lift": [2, "sts3215"],
                    "elbow_flex": [3, "sts3215"],
                    "wrist_flex": [4, "sts3215"],
                    "wrist_roll": [5, "sts3215"],
                    "gripper": [6, "sts3215"],
                },
            ),
        }
    )

    mock: bool = False 

    def _get_cameras_config():
        cameras_dict = {}
        
        # Get active RealSense cameras
        realsense_cams = realsense_camera.get_active_realsensecams()
        for i, cam_idx in enumerate(realsense_cams, 1):
            cameras_dict[f"realsense_{i}"] = IntelRealSenseCameraConfig(
                camera_index=cam_idx,
                fps=30,
                width=640,
                height=480,
            )
        
        # Get active USB cameras
        usb_cams = usb_camera.get_active_usbcams()
        for i, cam_idx in enumerate(usb_cams, 1):
            cameras_dict[f"usb_{i}"] = OpenCVCameraConfig(
                camera_index=cam_idx,
                fps=30,
                width=640,
                height=480,
            )
        
        return cameras_dict
    
    cameras: dict[str, CameraConfig] = field(
        default_factory=_get_cameras_config.__get__(None, object)
    )


