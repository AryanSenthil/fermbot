# fermia_camera.realsense_camera.publisher.py 

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

from .utils import (clear_previous_camera_keys, 
                   get_all_connected_realsense_cameras, 
                   run_camera)

def run_publisher():
    """Main publisher function that manages all connected cameras."""
    # First, clear all previous camera keys
    clear_previous_camera_keys()
    
    camera_threads = {}
    last_camera_count = 0
    
    while True:
        try:
            # Detect all connected cameras
            cameras = get_all_connected_realsense_cameras()
            
            # If camera count has changed, clear keys and restart everything
            if len(cameras) != last_camera_count:
                print(f"Camera count changed from {last_camera_count} to {len(cameras)}. Refreshing all camera data.")
                clear_previous_camera_keys()
                
                # Stop all existing threads to force them to restart
                for thread_key in list(camera_threads.keys()):
                    camera_threads.pop(thread_key, None)
                
                last_camera_count = len(cameras)
            
            if len(cameras) == 0:
                print("No cameras detected. Not publishing any data.")
                # Sleep longer when no cameras are connected to reduce log spam
                time.sleep(2)
                continue
                
            
            # Start threads for each camera that doesn't have an active thread
            for idx, camera in enumerate(cameras, 1):
                camera_key = f"camera_{camera['serial_number']}"
                
                if camera_key not in camera_threads or not camera_threads[camera_key].is_alive():
                    print(f"Starting thread for camera {idx} (S/N: {camera['serial_number']})")
                    thread = threading.Thread(
                        target=run_camera,
                        args=(camera['index'], camera['serial_number'], idx),
                        daemon=True
                    )
                    thread.start()
                    camera_threads[camera_key] = thread
            
            # Sleep to avoid excessive CPU usage
            time.sleep(0.03)
            
        except Exception as e:
            print("Exception in main publisher:", e)
            time.sleep(1)

if __name__ == "__main__":
    run_publisher()