# VideoCaptureThreading.py
# A multithreaded wrapper around cv2.VideoCapture
# Implementation adapted from https://github.com/gilbertfrancois/video-capture-async
import cv2
import threading

class VideoCaptureThreading:
    # Implementation adapted from https://github.com/gilbertfrancois/video-capture-async
    def __init__(self, src=0, api_preference=cv2.CAP_GSTREAMER):
        self.src = src
        self.cap = cv2.VideoCapture(self.src, api_preference)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.cap.release()  # Redundant because of self.__exit__() but here for compatibility reasons

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
