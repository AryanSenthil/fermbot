import time 
import threading 
import numpy as np
import cv2
import platform 
import shutil
import argparse 
from pathlib import Path
from PIL import Image

from threading import Thread 
import concurrent.futures

from fermbot.common.robot_devices.cameras.configs import OpenCVCameraConfig

from fermbot.common.robot_devices.utils import (
    RobotDeviceAlreadyConnectedError,
    RobotDeviceNotConnectedError,
    busy_wait
)

from fermbot.common.utils.utils import capture_timestamp_utc

import fermia_camera.usb_camera as usb_camera

class OpenCVCamera:
    """
    The OpenCVCamera class allows to efficiently record images from cameras. It uses the fermia_camera.usb_camera
    module which relies on Redis to communicate with the cameras.

    An OpenCVCamera instance requires a camera index (e.g. `OpenCVCamera(camera_index=1)`). When you only have one camera
    like a webcam of a laptop, the camera index is expected to be 1, but it might also be very different, and the camera index
    might change if you reboot your computer or re-plug your camera.

    To find the camera indices of your cameras, you can run our utility script that will be save a few frames for each camera:
    ```bash
    python fermbot/common/robot_devices/cameras/redis_camera.py --images-dir outputs/images_from_cameras
    ```

    Example of usage:
    ```python
    from fermbot.common.robot_devices.cameras.configs import OpenCVCameraConfig

    config = OpenCVCameraConfig(camera_index=1)
    camera = OpenCVCamera(config)
    camera.connect()
    color_image = camera.read()
    # when done using the camera, consider disconnecting
    camera.disconnect()
    ```

    Example of changing default fps, width, height and color_mode:
    ```python
    config = OpenCVCameraConfig(camera_index=1, fps=30, width=1280, height=720)
    config = OpenCVCameraConfig(camera_index=1, fps=90, width=640, height=480)
    config = OpenCVCameraConfig(camera_index=1, fps=90, width=640, height=480, color_mode="bgr")
    ```
    """

    def __init__(self, config: OpenCVCameraConfig):
        self.config = config 
        self.camera_index = config.camera_index 
        
        # Store the raw (capture) resolution from the config 
        self.capture_width = config.width 
        self.capture_height = config.height

        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps 
        self.channels = config.channels
        self.color_mode = config.color_mode 
        self.mock = config.mock 

        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.logs = {}
        
        # Set rotation parameters
        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def connect(self):
        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(f"OpenCVCamera({self.camera_index}) is already connected.")
        
        if self.mock:
            import tests.cameras.mock_cv2 as cv2
        else:
            import cv2 

        # Ensure the publisher is running before trying to connect 
        usb_camera.ensure_publisher() 
        time.sleep(5.0)

        max_retries = 40
        retry_delay = 0.5

        for attempt in range(max_retries):
            image = usb_camera.get_image(self.camera_index)

            if image is not None:
                break 

            if attempt < max_retries - 1:
                print(f"Camera {self.camera_index} not found yet. Waiting for publisher to detect camera.")
                time.sleep(retry_delay)

            else:
                raise ValueError(
                    f"Camera index {self.camera_index} is not availablle."
                )
            
        # Test that we can read an image
        img = usb_camera.get_image(self.camera_index)
        if img is None:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")
            
        # Set is_connected to True after successful connection
        self.is_connected = True
        
        # Get actual camera properties from the image
        h, w, _ = img.shape
        self.capture_height = h
        self.capture_width = w
        
        # Use the config FPS or default to 30
        self.fps = self.config.fps if self.config.fps is not None else 30


    def read(self, temporary_color_mode: str | None = None) -> np.ndarray:
        """
        Read a frame from the camera returned in the format (height, width, channels)
        (e.g. 480 x 640 x 3), contrarily to the pytorch format which is channel first.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )
        
        start_time = time.perf_counter()

        color_image = usb_camera.get_fermbot_image(camera_index=self.camera_index,
                                                    width=self.width,
                                                    height=self.height)
        
        if color_image is None:
            raise OSError(f"Can't capture color image from camera {self.camera_index}.")
        
        requested_color_mode = self.color_mode if temporary_color_mode is None else temporary_color_mode
    
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
        )

        # OpenCV uses BGR format as default (blue, green, red) for all operations, including displaying images.
        # However, Deep Learning framework such as fermbot uses RGB format as default to train neural networks,
        # so we convert the image color from BGR to RGB.
        if requested_color_mode == "rgb":
            if self.mock:
                import tests.cameras.mock_cv2 as cv2
            else:
                import cv2

            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            # Apply rotation if configured
        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)
            
        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time
        
        # log the utc time at which the image was received
        self.logs["timestamp_utc"] = time.time()
        
        self.color_image = color_image
        
        return color_image
    
    def read_loop(self):
        """Background thread function for async_read"""
        while not self.stop_event.is_set():
            try:
                self.color_image = self.read() 

                time.sleep(1/self.fps)
            except Exception as e:
                print(f"Error reading in thread:{e}")

    def async_read(self):
        """
        None-blocking version of read(). Starts a background thread if not already running. 
        Returns the most recent frame captured by the background thread.
        """

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()`"
            )
        
        if self.thread is None:
            self.stop_event = threading.Event() 
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True 
            self.thread.start() 

        num_tries = 0 
        while True:
            if self.color_image is not None:
                return self.color_image
                
            time.sleep(1 / self.fps)
            num_tries += 1
            if num_tries > self.fps * 2:
                raise TimeoutError("Timed out waiting for async_read() to start.")
            

    def disconnect(self):
        """Disconnect from the camera."""
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                f"OpenCVCamera({self.camera_index}) is not connected. Try running `camera.connect()` first."
            )
            
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()  # wait for the thread to finish
            self.thread = None
            self.stop_event = None
            
        self.is_connected = False

    def __del__(self):
        """Clean up resources when the object is garbage collected."""
        if getattr(self, "is_connected", False):
            self.disconnect()

def find_cameras():
    """
    Find all available cameras and return their information.
    """
    cameras = []
    
    for camera_index in usb_camera.get_active_usbcams():
        camera_data = usb_camera.get_usbcam_data(camera_index)
        cameras.append({
            "index": camera_index,
            "data": camera_data
        })
    
    return cameras

def save_image(img_array, camera_index, frame_index, images_dir):
    img = Image.fromarray(img_array)
    path = images_dir / f"camera_{camera_index:02d}_frame_{frame_index:06d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path), quality=100)


def save_images_from_cameras(
        images_dir: Path, 
        camera_ids: list | None = None, 
        fps=None, 
        width=None, 
        height=None, 
        record_time_s=2,
        mock=False 
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera 
    associated to a given camera index.
    """

    if camera_ids is None or len(camera_ids) == 0:
        camera_infos = find_cameras()
        camera_ids = [cam["index"] for cam in camera_infos]

    print("Connecting Cameras")

    cameras = []
    for cam_idx in camera_ids:
        config = OpenCVCameraConfig(camera_index=cam_idx, fps=fps, width=width, height=height, mock=mock)
        camera = OpenCVCamera(config)
        camera.connect()
        print(
            f"OpenCVCamera({camera.camera_index}, fps={camera.fps}, width={camera.capture_width}, "
            f"height={camera.capture_height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok = True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter() 

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        while True:
            now = time.perf_counter() 

            for camera in cameras:
                image = camera.read() if fps is None else camera.async_read()

                executor.submit(
                    save_image,
                    image,
                    camera.camera_index,
                    frame_index,
                    images_dir,
                )

            if fps is not None:
                dt_s = time.perf_counter() - now
                busy_wait(1 / fps - dt_s)

            print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

            if time.perf_counter() - start_time > record_time_s:
                break

            frame_index += 1

    print(f"Images have been saved to {images_dir}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `OpenCVCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--camera-ids",
        type=int,
        nargs="*",
        default=None,
        help="List of camera indices used to instantiate the `OpenCVCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_opencv_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=4.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))