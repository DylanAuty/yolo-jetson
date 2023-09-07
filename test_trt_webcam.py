# test_trt_webcam.py
# Testing script for running Yolov7 inference on an h264-encoded rtp stream on port 5000.
# Displays the results as a video in an x window, so must be run with $DISPLAY set
import argparse
import os
import cv2
import time
import json

import yolojetson.utils
from yolojetson.VideoCaptureThreading import VideoCaptureThreading
from yolojetson.TRTBaseEngine import TRTBaseEngine

most_recent_results = {}

def main(args):
    if args.save_video:
        video_save_dir = os.path.join("saved_runs", f"capture_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
        if not os.path.exists(video_save_dir):
            os.makedirs(video_save_dir)
        print(f"Saving video capture to {video_save_dir}")

    print("Setting up video stream")
    video = VideoCaptureThreading('\
            udpsrc port=5000 \
            ! application/x-rtp,clock-rate=90000,encoding-name=H264,payload=96 \
            ! rtpjitterbuffer latency=1000 \
            ! rtph264depay \
            ! queue \
            ! avdec_h264 \
            ! videoconvert \
            ! video/x-raw,format=BGR \
            ! appsink sync=false \
            ', cv2.CAP_GSTREAMER)
    print("Setting up prediction engine")
    pred = TRTBaseEngine(engine_path=args.checkpoint, imgsz=(args.resolution[0],args.resolution[1]))

    # Main capture/prediction/display loop.
    print("Starting capture loop")
    video.start()
    start_time = None
    global most_recent_results
    while True:
        start_time = time.time()
        ret, image = video.read()
        origin_img, most_recent_results = pred.inference_image(image, do_visualise=True)
        most_recent_results['timestamp'] = start_time
        if args.save_video:
            filename = os.path.join(video_save_dir, f"{str(start_time)}.png")
            cv2.imwrite(filename, origin_img)
        cv2.imshow('frame', origin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.setWindowTitle('frame', f'FPS: {1 / (time.time() - start_time):4.2f}')
        json_out = yolojetson.utils.detection_to_json(most_recent_results, class_names=pred.class_names)
        print(json_out)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', default='./checkpoints/yolov7_640-nms.trt', help='Path to the tensorrt checkpoint/engine to use.')
    parser.add_argument('--resolution', '-r', type=str, default="640,640", help='Resolution of video as a comma separated list (e.g. "width,height"). Should normally be square (640,640 or 1280,1280).')
    parser.add_argument('--save_video', action='store_true', help='Save the annotated video frames to a new output directory in saved_runs')

    args = parser.parse_args()
    args.resolution = [int(item) for item in args.resolution.split(',')]
    main(args)
