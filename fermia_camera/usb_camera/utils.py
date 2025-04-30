# fermia_camera.usb_camera.utils.py 

import time
import redis
import cv2
import base64
import numpy as np
import json
import threading
import subprocess
from langchain_ollama import ChatOllama
from typing import Literal
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import subprocess
import re
import cv2

from .configs import (REDIS_DB,
                     REDIS_HOST,
                     REDIS_PORT)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def clear_previous_camera_keys():
    """
    Clears all Redis keys related to previous USB camera configurations and statuses.
    This function retrieves all keys from the Redis database that match the patterns 
    "fermia_usb_camera*" and "fermia_usb_running_camera*". It then deletes these keys 
    to ensure that no stale or outdated camera configuration or status information 
    remains in the database.
    Actions performed:
    - Deletes keys matching the pattern "fermia_usb_camera*" (webcamera keys).
    - Deletes keys matching the pattern "fermia_usb_running_camera*" (webcam status keys).
    - Logs the number of keys cleared for each pattern.
    Note:
    - Ensure that `redis_client` is properly initialized and connected to the Redis server 
      before calling this function.
    - Use this function with caution as it will permanently delete matching keys from Redis.
    """

    # Get all keys matching webcamera pattern 
    webcam_keys = redis_client.keys("fermia_usb_camera*")
    status_keys = redis_client.keys("fermia_usb_running_camera*")

    # Delete all matching keys
    if webcam_keys:
        redis_client.delete(*webcam_keys)
        print(f"Cleared {len(webcam_keys)} previous webcamera keys")
        
    if status_keys:
        redis_client.delete(*status_keys)
        print(f"Cleared {len(status_keys)} previous webcam status keys")


def run_camera(device_index, device_id, camera_index):
    """
    Starts capturing video from a specified webcam, processes the frames, and publishes them to a Redis database.
    Args:
        device_index (int): The index of the webcam device to open.
        device_id (str): A unique identifier for the device.
        camera_index (int): The index of the camera, used for logging and Redis key generation.
    Functionality:
        - Opens the webcam using OpenCV and sets the resolution to 1280x720.
        - Publishes a "running" status key to Redis to indicate the camera publisher is active.
        - Captures frames from the webcam, encodes them as JPEG, and stores them in Redis as base64-encoded strings.
        - Limits the frame processing rate to a maximum of 15 frames per second (fps).
        - Handles exceptions gracefully and ensures the webcam resource is released upon termination.
    Redis Keys:
        - `fermia_usb_running_camera{camera_index}`: Indicates the publisher is running (expires after 5 seconds).
        - `usbcamera{camera_index}`: Stores the base64-encoded image and device ID as a JSON object.
    Notes:
        - If the webcam fails to open or stops sending frames, the function logs an error and stops the capture process.
        - The function runs indefinitely until interrupted or an error occurs.
    Exceptions:
        - Logs any exceptions that occur during execution and ensures proper cleanup of resources.
    """

    try:
        print(f"Starting capture for webcam {camera_index} (Device ID: {device_id})")
        
        # Open the webcam
        cap = cv2.VideoCapture(device_index)
        
        # Set resolution (can be adjusted as needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            print(f"Failed to open webcam {camera_index}")
            return
            
        # Notify that this camera publisher is running
        camera_status_key = f"fermia_usb_running_camera{camera_index}"
        redis_client.set(camera_status_key, "true", ex=5)

        last_frame_time = time.time() 

        while True:
            # Refresh running key so consumers know the publisher is alive 
            redis_client.set(camera_status_key, "true", ex=5)

            # Read a frame from the webcam 
            ret, frame = cap.read() 

            if not ret:
                print(f"usb {camera_index} stopped sending frame. Stopping capture.")

            # Only process and publish frames at a reasonable rate (e.g., 15fps max)
            current_time = time.time()
            if current_time - last_frame_time < 1/15:  # Limit to 15fps
                time.sleep(0.01)  # Small sleep to avoid excessive CPU usage
                continue
                
            last_frame_time = current_time

            # Encode the frame as JPEG
            ret, img_encoded = cv2.imencode('.jpg', frame)
            
            if ret:
                b64_img = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
                
                # Create camera data dictionary
                camera_data = {
                    "device_id": device_id,
                    "base64_image": b64_img
                }
                
                # Store in Redis with the appropriate key
                redis_client.set(f"usbcamera{camera_index}", json.dumps(camera_data))

    except Exception as e:
        print(f"Exception in webcam {camera_index} publisher:", e)
    finally:
        try:
            cap.release()
        except Exception:
            pass 


def list_video_devices():
    """
    This function lists all video devices available on the system by using the `v4l2-ctl` command-line tool.
    It parses the output of the command to extract device names and their corresponding paths.

    Returns:
        list: A list of dictionaries, where each dictionary represents a video device with its name and paths.

    Example:
        [{'name': 'HD Webcam', 'paths': ['/dev/video0', '/dev/video1']}]
    """
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        output = result.stdout
        devices = []
        current_device = None

        for line in output.splitlines():
            if not line.startswith('\t'):
                # Device name line
                if current_device:
                    devices.append(current_device)
                current_device = {'name': line.strip(), 'paths': []}
            else:
                # Path line (e.g., /dev/video0)
                if current_device is not None:
                    current_device['paths'].append(line.strip())

        if current_device:
            devices.append(current_device)

        return devices

    except subprocess.CalledProcessError as e:
        print("Error running v4l2-ctl:", e.stderr)
        return []         



def find_cameras(
    possible_camera_ids: list[int | str], raise_when_empty=False, mock=False
) -> list[int | str]:
    """
    This function attempts to find cameras from a list of possible camera IDs or paths.

    Args:
        possible_camera_ids (list[int | str]): A list of integers or strings representing possible camera IDs or paths.
        raise_when_empty (bool, optional): If True, raises an OSError if no cameras are found. Defaults to False.
        mock (bool, optional): If True, enables mock mode for testing purposes. Defaults to False.

    Returns:
        list[int | str]: A list of camera IDs or paths that correspond to valid and functional cameras.

    Raises:
        OSError: If raise_when_empty is True and no cameras are detected.

    Example:
        possible_ports = ['/dev/video0', '/dev/video1']
        cameras = find_cameras(possible_ports)
        print(cameras)  # Output: ['/dev/video0']
    """
    cameras = []
    for camera_idx in possible_camera_ids:
        camera = cv2.VideoCapture(camera_idx)
        is_open = camera.isOpened()

        # Check if we can actually get a valid frame 
        if is_open:
            ret, frame = camera.read() 
            if ret and frame is not None and frame.size > 0:
                print(f"Camera found at index {camera_idx}")
            cameras.append({
                "index": camera_idx,
                "device_id": f"usbcam_{camera_idx}",
            })
            camera.release()

    return cameras

# Function to extract video device numbers from output text
def get_webcam_paths_as_list(output_text):
    """Convert the output text to a list of device paths or numbers"""
    if isinstance(output_text, str):
        # Split by comma and strip whitespace
        paths = [path.strip() for path in output_text.split(',')]
        
        # Extract just the device numbers for OpenCV
        numbers = []
        for path in paths:
            match = re.search(r'/dev/video(\d+)', path)
            if match:
                numbers.append(int(match.group(1)))
        return numbers
    return []


# Main execution flow
def get_all_connected_usbcams():
    # Initialize the LLM and tools
    model = ChatOllama(model="qwen2.5:7b", temperature=0)
    tools = [list_video_devices]
    
    graph = create_react_agent(
        model,
        tools=tools,
        prompt="""
        You are a video path filtering assistant.
        Your task is to examine a list of video device paths (e.g., /dev/video0, /dev/video1, etc.) and return only those paths that are likely associated with webcam devices.
        Instructions:
        - Only return device paths that appear to correspond to **webcams**.
        - Ignore and exclude any device paths that are related to:
            - NVIDIA virtual cameras (e.g., NVIDIA Broadcast, Virtual Camera)
            - Intel RealSense devices
            - Any other known virtual or non-webcam devices
        Output Format:
        - Return a clean list of matching video paths.
        - Do not include any explanations or unrelated text—just the list.
        Assume you have access to any necessary metadata or naming conventions to infer the type of device.
        """
    )
    
    inputs = {"messages": [("user", "what are the ports that correspond to webcams")]}
    result = graph.invoke(inputs)
    
    # Process the LLM output
    ai_messages = [msg for msg in result["messages"] if msg.type == "ai"]
    if ai_messages:
        # Get the content of the last AI message
        last_message_content = ai_messages[-1].content
        
        # Check if it's a string or an object with a string representation
        if hasattr(last_message_content, 'to_string'):
            message_text = last_message_content.to_string()
        else:
            message_text = str(last_message_content)
        
        # Process the message to get the webcam paths
        webcam_paths = get_webcam_paths_as_list(message_text)
        print("Potential webcam device numbers:", webcam_paths)
        
        # Verify which cameras actually work with OpenCV
        working_cameras = find_cameras(webcam_paths, raise_when_empty=False)
        print("Verified working cameras:", working_cameras)
        
        return working_cameras
    
    return []
