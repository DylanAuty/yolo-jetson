import sys
import cv2
import time
import threading

class VideoCaptureThreading:
    # Implementation adapted from https://github.com/gilbertfrancois/video-capture-async
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
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

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()


if __name__ == "__main__":
    # Gstreamer pipeline to read RTP packets from port 5000, that it assumes are h264-encoded.
    print("Setting up stream")

    # Using multi-threaded capture; replaces cv2.VideoCapture.
    # If using multithreaded, add vid.start() and vid.stop() before and after the main capture loop.
    vid = VideoCaptureThreading('\
            udpsrc port=5000 \
            ! application/x-rtp,encoding-name=H264,payload=96 \
            ! rtph264depay \
            ! avdec_h264 \
            ! videoconvert \
            ! video/x-raw,format=BGR \
            ! appsink sync=false drop=true \
            ', cv2.CAP_GSTREAMER)

    if not vid.isOpened():
        sys.exit("Error: could not open stream")
    else:
        print("Successfully opened stream. Reading...")

    vid.start()
    while True:
        start_time = time.time()
        ret, frame = vid.read()
        if not ret:
            print("Failed to read from video stream.")
            continue
        if frame is not None:
            cv2.imshow(f'Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Frame read but is still None (gstreamer settings are likely wrong).")

    vid.stop()
    vid.release()
    cv2.destroyAllWindows()
