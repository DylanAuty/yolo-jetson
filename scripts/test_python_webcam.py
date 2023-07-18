import sys
import cv2
import time
import threading

from modules import VideoCaptureThreading

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
