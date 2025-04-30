import abc 
from dataclasses import dataclass 

import draccus 

@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
    

@CameraConfig.register_subclass("opencv")
@dataclass 
class OpenCVCameraConfig(CameraConfig):
    """
    Example of tested options for OpenCV USB Camera:

    ```python
    OpenCVCameraConfig(6, 30, 640, 480)
    OpenCVCameraConfig(6, 60, 640, 480)
    OpenCVCameraConfig(6, 90, 640, 480)
    OpenCVCameraConfig(6, 30, 1280, 720)
    ```
    """
    camera_index: int
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")
        
@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(332522072818, 30, 640, 480)
    IntelRealSenseCameraConfig(332522072818, 60, 640, 480)
    IntelRealSenseCameraConfig(332522072818, 90, 640, 480)
    IntelRealSenseCameraConfig(332522072818, 30, 1280, 720)
    IntelRealSenseCameraConfig(332522072818, 30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(332522072818, 30, 640, 480, rotation=90)
    ```
    """

    camera_index: int
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")