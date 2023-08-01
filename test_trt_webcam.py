# test_trt_webcam.py
# Testing script for running Yolov7 inference on an h264-encoded rtp stream on port 5000.
# Displays the results as a video in an x window, so must be run with $DISPLAY set
import argparse
import cv2
import time
import json

from yolojetson.VideoCaptureThreading import VideoCaptureThreading
from yolojetson.TRTBaseEngine import TRTBaseEngine

most_recent_results = {}

def main(args):
    print("Setting up video stream")
    video = VideoCaptureThreading('\
            udpsrc port=5000 \
            ! application/x-rtp,encoding-name=H264,payload=96 \
            ! rtph264depay \
            ! avdec_h264 \
            ! videoconvert \
            ! video/x-raw,format=BGR \
            ! appsink sync=false drop=true \
            ', cv2.CAP_GSTREAMER)
    print("Setting up prediction engine")
    pred = TRTBaseEngine(engine_path=args.checkpoint, imgsz=(640,640))

    # Main capture/prediction/display loop.
    print("Starting capture loop")
    video.start()
    start_time = None
    global most_recent_results
    while True:
        start_time = time.time()
        ret, image = video.read()
        origin_img, most_recent_results = pred.inference_image(image)
        most_recent_results['time'] = start_time
        cv2.imshow('frame', origin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.setWindowTitle('frame', f'FPS: {1 / (time.time() - start_time):4.2}')
        print(json.dumps(most_recent_results, indent=4))

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default='./checkpoints/yolov7_640-nms.trt', help='Path to the tensorrt checkpoint/engine to use.')

    args = parser.parse_args()
    main(args)
