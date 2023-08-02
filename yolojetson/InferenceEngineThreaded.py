# InferenceEngineThreaded.py
# Threaded implementation of the inference engine.
import cv2
import threading
import time
import copy

import yolojetson.utils
from yolojetson.VideoCaptureThreading import VideoCaptureThreading
from yolojetson.TRTBaseEngine import TRTBaseEngine

class InferenceEngineThreaded:
    def __init__(self, video_src=0, video_api_pref=cv2.CAP_GSTREAMER, checkpoint='./checkpoints/yolov7_640-nms.trt'):
        """
        Captures video and runs YOLO inference asynchronously.
        
        :param video_src: cv2 video source. Can be an int or a string.
        :param video_api_pref: cv2 VideoCapture api preference. cv2.CAP_GSTREAMER by default.
        :param checkpoint: Path to TensorRT checkpoint/engine file.
        """
        self.video = VideoCaptureThreading(video_src, video_api_pref)
        self.engine = TRTBaseEngine(engine_path=checkpoint, imgsz=(640, 640))
        self.detections = {}
        self.annotated_image = None

        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            print('[!] Threaded inference engine already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            ret, image = self.video.read()
            timestamp = time.time()
            annotated_image, detections = self.engine.inference_image(image)
            detections['timestamp'] = timestamp
            detections = yolojetson.utils.detection_to_json(detections, class_names=self.engine.class_names)
            with self.read_lock:
                self.detections = detections
                self.annotated_image = annotated_image

    def read(self):
        with self.read_lock:
            annotated_image = copy.deepcopy(self.annotated_image)
            detections = copy.deepcopy(self.detections)
        return detections, annotated_image

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        return
