# fermia_camera.usb_camera.publisher.py 

import time
import redis
import cv2
import base64
import numpy as np
import json
import threading

from .configs import (REDIS_DB,
                     REDIS_HOST,
                     REDIS_PORT)

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


from .utils import (clear_previous_camera_keys,
                   get_all_connected_usbcams,
                   run_camera)

def run_publisher():
    clear_previous_camera_keys()

    camera_threads = {}
    last_camera_count = 0 
    cameras = get_all_connected_usbcams()

    while True:
        try:
            if len(cameras) == 0:
                print("No webcams detected. Not publishing any data")
                time.sleep(2)
                continue


            # Start thread for each camer that doesn't have an active thread 
            for idx, camera in enumerate(cameras, 1):
                camera_key = f"usbcam_{camera['index']}"

                if camera_key not in camera_threads or not camera_threads[camera_key].is_alive():
                    print(f"Starting thread for webcam {idx} (Device ID: {camera['device_id']})")
                    thread = threading.Thread(
                        target=run_camera,
                        args=(camera['index'], camera['device_id'], idx),
                        daemon=True
                    )
                    thread.start()
                    camera_threads[camera_key] = thread
            time.sleep(0.03)


        except Exception as e:
            print("Exception in main publisher:", e)
            time.sleep(1)



if __name__ == "__main__":
    run_publisher()