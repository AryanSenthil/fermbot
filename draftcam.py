# import pyrealsense2 as rs
# import numpy as np
# import cv2

# def capture_realsense_depth():
#     # Configure depth and color streams
#     pipeline = rs.pipeline()
#     config = rs.config()
    
#     # Enable depth stream
#     config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
#     # Start streaming
#     pipeline.start(config)
    
#     # Create colorizer for visualizing depth data
#     colorizer = rs.colorizer()
    
#     print("RealSense depth feed started")
#     print("Press 'q' to quit")
    
#     try:
#         while True:
#             # Wait for a coherent pair of frames
#             frames = pipeline.wait_for_frames()
#             depth_frame = frames.get_depth_frame()
            
#             if not depth_frame:
#                 continue
                
#             # Convert depth frame to numpy array
#             depth_image = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            
#             # Display the depth image
#             cv2.imshow('RealSense Depth Feed', depth_image)
            
#             # Break the loop if 'q' is pressed
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     finally:
#         # Stop streaming
#         pipeline.stop()
#         cv2.destroyAllWindows()
#         print("RealSense depth feed stopped")

# if __name__ == "__main__":
#     capture_realsense_depth()

import cv2

def capture_from_webcam():
    # Create a VideoCapture object
    # 0 is usually the default camera (built-in webcam)
    # You can try different indices (1, 2, etc.) if you have multiple cameras
    cap = cv2.VideoCapture(4, 6)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set the resolution to 1280x720
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # # Verify the resolution was set correctly
    # actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(f"Webcam resolution set to: {actual_width}x{actual_height}")
    
    print("Webcam opened successfully!")
    print("Press 'q' to quit")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # If frame is not successfully captured, break the loop
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Display the resulting frame
        cv2.imshow('Webcam Feed', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture and destroy windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_from_webcam()