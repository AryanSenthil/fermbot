# fermia_camera.realsense_camera.utils.py 

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
import pyrealsense2 as rs 
import os 


from .configs import (REDIS_DB,
                     REDIS_HOST,
                     REDIS_PORT)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def ensure_publisher():
    """
    Checks if the publisher is running (via a Redis key). If not, spawns it.
    """
    # Check if any camera publisher is running by searching for any camera status keys
    publisher_running = False
    for key in redis_client.scan_iter("fermia_realsense_running_camera*"):
        publisher_running = True
        break
    
    if not publisher_running:
        # Attempt to acquire a lock to avoid simultaneous spawns
        lock_acquired = redis_client.set("fermia_realsense_publisher_lock", str(os.getpid()), nx=True, ex=10)
        if lock_acquired:
            print("No publisher running. Starting publisher automatically.")
            # Spawn the publisher process in the background
            subprocess.Popen(["python", "-m", "fermia_camera.realsense_camera.publisher"])
            # Allow a moment for the publisher to initialize
            time.sleep(2)

# Ensure the publisher is running when the package is imported.
ensure_publisher()

def get_active_realsensecams():
    """
    Returns a list of active camera indices.
    """
    cameras = []
    for key in redis_client.scan_iter("realsensecamera*"):
        camera_index = key.decode('utf-8').replace('realsensecamera', '')
        cameras.append(int(camera_index))
    return sorted(cameras)

def get_camera_data(camera_index=1):
    """
    Retrieves the complete camera data (serial number, color image, depth image) for a specific camera.
    Returns:
        Dictionary with camera data or None if unavailable.
    """
    camera_data_json = redis_client.get(f"realsensecamera{camera_index}")
    if camera_data_json is None:
        return None
    
    try:
        return json.loads(camera_data_json)
    except Exception:
        return None

def get_image(camera_index=1):
    """
    Retrieves the latest image from Redis for a specific camera.
    Returns:
        The decoded OpenCV image (numpy array) or None if unavailable.
    """
    camera_data = get_camera_data(camera_index)
    if camera_data is None or 'base64_image' not in camera_data:
        return None
    
    try:
        b64_img = camera_data['base64_image']
        img_bytes = base64.b64decode(b64_img)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None

def get_base64_image(camera_index=1):
    """
    Retrieves the latest base64 image from Redis for a specific camera.
    Returns:
        The base64 image string or None if unavailable.
    """
    camera_data = get_camera_data(camera_index)
    if camera_data is None or 'base64_image' not in camera_data:
        return None
    
    return camera_data['base64_image']

def get_depth_data(camera_index=1):
    """
    Retrieves the latest depth array from Redis for a specific camera.
    Returns:
        The depth array as a numpy array or None if unavailable.
    """
    camera_data = get_camera_data(camera_index)
    if camera_data is None or 'depth_image' not in camera_data:
        return None
    
    try:
        b64_depth = camera_data['depth_image']
        depth_bytes = base64.b64decode(b64_depth)
        # Convert the raw bytes back into numpy array
        depth_array = np.frombuffer(depth_bytes, dtype=np.uint16)
        depth_array = depth_array.reshape((720, 1280))
        return depth_array
    except Exception:
        return None

def get_depth_image(camera_index=1):
    """
    Retrieves the latest depth image from Redis for a specific camera.
    Returns:
        The colorized depth image or None if unavailable.
    """
    depth_array = get_depth_data(camera_index)
    if depth_array is None:
        return None
    
    try:
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_array, alpha=0.06), cv2.COLORMAP_JET)
        return depth_colormap
    except Exception:
        return None

def get_camera_serial_number(camera_index=1):
    """
    Retrieves the serial number for a specific camera.
    Returns:
        The serial number string or None if unavailable.
    """
    camera_data = get_camera_data(camera_index)
    if camera_data is None or 'serial_number' not in camera_data:
        return None
    
    return camera_data['serial_number']

def get_all_cameras_data():
    """
    Retrieves data for all connected cameras.
    Returns:
        Dictionary mapping camera indices to their data.
    """
    cameras = {}
    for camera_index in get_active_realsensecams():
        camera_data = get_camera_data(camera_index)
        if camera_data:
            cameras[camera_index] = camera_data
    
    return cameras


def get_fermbot_image(camera_index=1, width=640, height=480):
    """
    Retrieves the latest image from Redis and resizes it to the given dimensions.
    
    Args:
        camera_index (int): The index of the camera.
        width (int): Desired image width.
        height (int): Desired image height.
    
    Returns:
        Resized OpenCV image (numpy array) or None if unavailable.
    """
    img = get_image(camera_index)
    if img is None:
        return None
    try:
        resized_img = cv2.resize(img, (width, height))
        return resized_img
    except Exception:
        return None