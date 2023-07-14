import cv2
import time

# Gstreamer pipeline to read RTP packets from port 5000, that it assumes are h264-encoded.
print("Setting up stream")

vid = cv2.VideoCapture('''
        udpsrc port=5000 
        ! application/x-rtp,encoding-name=H264,payload=96 
        ! rtph264depay 
        ! avdec_h264
        ! videoconvert 
        ! appsink sync=false drop=true
        ''', cv2.CAP_GSTREAMER)

print("Roll until frame is received")
frame = None

print("Got frame. Displaying stream.")
while True:
    start_time = time.time()
    ret, _ = vid.read(frame)
    if frame is None:
        continue

    cv2.imshow(f'Frame {1 / (start_time - time.time())}', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
