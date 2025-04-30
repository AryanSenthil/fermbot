# fermia_camera.usb_camera.__init__.py 
import redis
import subprocess
import os
import time
import base64
import cv2
import numpy as np
import json

from .configs import (REDIS_DB,
                     REDIS_HOST,
                     REDIS_PORT)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def ensure_publisher():
    """
    Checks if the usbcam publisher is running (via Redis keys). If not, spawns it.
    """
    # Check if any usbcam publisher is running
    publisher_running = False
    for key in redis_client.scan_iter("fermia_usb_running_camera*"):
        publisher_running = True
        break
    
    if not publisher_running:
        # Attempt to acquire a lock to avoid simultaneous spawns
        lock_acquired = redis_client.set("fermia_usbcam_publisher_lock", str(os.getpid()), nx=True, ex=10)
        if lock_acquired:
            print("No usbcam publisher running. Starting publisher automatically.")
            # Spawn the publisher process in the background
            subprocess.Popen(["python", "-m", "usb_camera.publisher"])
            # Allow a moment for the publisher to initialize
            time.sleep(2)

# Ensure the publisher is running when the package is imported.
ensure_publisher()

def get_usbcam_data(camera_index=1):
    """
    Retrieves the complete usbcam data for a specific camera.
    Returns:
        Dictionary with usbcam data or None if unavailable.
    """
    camera_data_json = redis_client.get(f"usbcamera{camera_index}")
    if camera_data_json is None:
        return None
    
    try:
        return json.loads(camera_data_json)
    except Exception:
        return None
    

def get_image(camera_index=1):
    """
    Retrieves the latest image from Redis for a specific usbcam.
    Returns:
        The decoded OpenCV image (numpy array) or None if unavailable.
    """
    camera_data = get_usbcam_data(camera_index)
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
    Retrieves the latest base64 image from Redis for a specific usbcam.
    Returns:
        The base64 image string or None if unavailable.
    """
    camera_data = get_usbcam_data(camera_index)
    if camera_data is None or 'base64_image' not in camera_data:
        return None
    
    return camera_data['base64_image']


def get_active_usbcams():
    """
    Returns a list of active webcam indices.
    """
    cameras = []
    for key in redis_client.scan_iter("usbcamera*"):
        camera_index = key.decode('utf-8').replace('usbcamera', '')
        cameras.append(int(camera_index))
    return sorted(cameras)


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

def get_all_usbcams_data():
    """
    Retrieves data for all connected usbcams.
    Returns:
        Dictionary mapping camera indices to their data.
    """
    cameras = {}
    for camera_index in get_active_usbcams():
        camera_data = get_usbcam_data(camera_index)
        if camera_data:
            cameras[camera_index] = camera_data
    
    return cameras