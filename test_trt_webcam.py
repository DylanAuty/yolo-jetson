import cv2
import time

import yolojetson.utils
import yolojetson.constants
from yolojetson.VideoCaptureThreading import VideoCaptureThreading
from yolojetson.TRTBaseEngine import TRTBaseEngine


def main():
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
    pred = TRTBaseEngine(engine_path='./checkpoints/yolov7_640-nms.trt', imgsz=(360,640))
    print("Starting capture loop")
    video.start()
    start_time = None
    while True:
        start_time = time.time()
        ret, image = video.read()
        origin_img = pred.inference_image(image)
        cv2.imshow('frame', origin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.setWindowTitle('frame', f'FPS: {1 / (time.time() - start_time):4.2}')
    pred.get_fps()
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
