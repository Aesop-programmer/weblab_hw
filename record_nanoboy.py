import cv2
import argparse
import multiprocessing as mp
import mediapipe as mp_2
import time
import numpy as np
def gstreamer_camera(queue):
    # Use the provided pipeline to construct the video capture in opencv
    pipeline = (
        "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)1920, height=(int)1080, "
            "format=(string)NV12, framerate=(fraction)30/1 ! "
        "queue ! "
        "nvvidconv flip-method=2 ! "
            "video/x-raw, "
            "width=(int)1920, height=(int)1080, "
            "format=(string)BGRx, framerate=(fraction)30/1 ! "
        "videoconvert ! "
            "video/x-raw, format=(string)BGR ! "
        "appsink"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    while True:
        ret, frame = cap.read()
        
        ### object detection
        mp_object_detection = mp_2.solutions.object_detection
        mp_drawing = mp_2.solutions.drawing_utils     
        
        with mp_object_detection.ObjectDetection(
            min_detection_confidence=0.1) as object_detection:

            results = object_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)
        ### hand pose tracking
        mp_hands = mp.solutions.hands
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_drawing = mp.solutions.drawing_utils

        with mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
        ### 
        
        if not ret:
            break
        print(type(frame))
        queue.put(frame)
    # Complete the function body
    pass


def gstreamer_rtmpstream(queue):
    # Use the provided pipeline to construct the video writer in opencv
    pipeline = (
        "appsrc ! "
            "video/x-raw, format=(string)BGR ! "
        "queue ! "
        "videoconvert ! "
            "video/x-raw, format=RGBA ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=8000000 ! "
        "h264parse ! "
        "flvmux ! "
        'rtmpsink location="rtmp://localhost/rtmp/live live=1"'
    )
    video_writer = cv2.VideoWriter(
        pipeline,
        cv2.CAP_GSTREAMER,
        30.0,
        (1920,1080),
        
    )
    while True:
        if (not queue.empty()):
            print("abc")
            video_writer.write(queue.get())
    video_writer.release()
    # Complete the function body
    # You can apply some simple computer vision algorithm here
    
    pass

if __name__ == '__main__':
    queue = mp.Queue(maxsize=1)
    pro = mp.Process(target=gstreamer_camera,args=(queue,))
    #con = mp.Process(target=consumer,args=(queue,))
    print("a")
    #con.start()
    pro.start()
    
    pipeline = (
        "appsrc ! "
            "video/x-raw, format=(string)BGR ! "
        "queue ! "
        "videoconvert ! "
            "video/x-raw, format=RGBA ! "
        "nvvidconv ! "
        "nvv4l2h264enc bitrate=8000000 ! "
        "h264parse ! "
        "flvmux ! "
        'rtmpsink location="rtmp://localhost/rtmp/live live=1"'
    )
    video_writer = cv2.VideoWriter(
        pipeline,
        cv2.CAP_GSTREAMER,
        30.0,
        (1920,1080),
        
    )
    while True:
        if (not queue.empty()):
            print("abc")
            video_writer.write(queue.get())
    time.sleep(10.0)
    pro.terminate()
    video_writer.release()
# Complelte the code
