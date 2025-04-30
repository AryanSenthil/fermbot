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


from .configs import (REDIS_DB,
                     REDIS_HOST,
                     REDIS_PORT)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)

def clear_previous_camera_keys():
    """Clear all previous realsense camera and status keys from Redis."""
    # Get all keys matching realsensecamera pattern
    camera_keys = redis_client.keys("realsensecamera*")
    status_keys = redis_client.keys("fermia_publisher_running_camera*")
    
    # Delete all matching keys
    if camera_keys:
        redis_client.delete(*camera_keys)
        print(f"Cleared {len(camera_keys)} previous realsense camera keys")
        
    if status_keys:
        redis_client.delete(*status_keys)
        print(f"Cleared {len(status_keys)} previous realsense camera status keys")

def get_all_connected_realsense_cameras():
    """Detect all connected RealSense cameras and return their serial numbers."""
    ctx = rs.context()
    devices = ctx.query_devices()
    cameras = []
    
    for dev_idx in range(len(devices)):
        device = devices[dev_idx]
        serial_number = device.get_info(rs.camera_info.serial_number)
        cameras.append({
            "index": dev_idx,
            "serial_number": serial_number
        })
    
    return cameras

def run_camera(device_index, serial_number, camera_index):
    """Run a publisher for a specific camera."""
    try:
        print(f"Starting pipeline for camera {camera_index} (S/N: {serial_number})")
        
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Select specific device by serial number
        config.enable_device(serial_number)
        
        # color image stream
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        # depth image stream
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
        
        pipeline.start(config)
        
        # Notify that this camera publisher is running
        camera_status_key = f"fermia_realsense_running_camera{camera_index}"
        redis_client.set(camera_status_key, "true", ex=5)
        
        while True:
            # Refresh running key so consumers know the publisher is alive
            redis_client.set(camera_status_key, "true", ex=5)
            
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except Exception:
                print(f"Camera {camera_index} stopped sending frames. Stopping pipeline.")
                break
                
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue
                
            # Process color frame
            img = np.asanyarray(color_frame.get_data())
            ret, img_encoded = cv2.imencode('.jpg', img)
            
            if ret:
                b64_img = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
                
                # Process depth frame
                depth_img = np.asanyarray(depth_frame.get_data())
                b64_depth = base64.b64encode(depth_img.tobytes()).decode('utf-8')
                
                # Create camera data dictionary
                camera_data = {
                    "serial_number": serial_number,
                    "base64_image": b64_img,
                    "depth_image": b64_depth
                }
                
                # Store in Redis with the appropriate key
                redis_client.set(f"realsensecamera{camera_index}", json.dumps(camera_data))
                
    except Exception as e:
        print(f"Exception in camera {camera_index} publisher:", e)
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
