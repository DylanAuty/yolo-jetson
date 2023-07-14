import cv2
import time

#vid = cv2.VideoCapture(0)
#vid = cv2.VideoCapture('udpsrc host=155.198.116.176 port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
#vid = cv2.VideoCapture('udpsrc port=5000 caps = "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)JPEG, payload=(int)26" ! rtpjpegdepay ! jpegdec ! videoconvert ! appsink', cv2.CAP_GSTREAMER)
#vid = cv2.VideoCapture('udpsrc uri=udp://155.198.116.176:5000 auto-multicast=true ! application/x-rtp, media=video, encoding
#vid = cv2.VideoCapture('udp://127.0.0.1:5000', cv2.CAP_FFMPEG)
#vid = cv2.VideoCapture('udp://155.198.116.176:5000', cv2.CAP_FFMPEG)
vid = cv2.VideoCapture('udpsrc port=5000 ! application/x-rtp,encoding-name=H264,payload=06 ! rtph264depay ! avdec_h264 ! videoconvert ! appsink', cv2.CAP_GSTREAMER)

print("Starting")
frame = None
while frame is None:
    ret, frame = vid.read()

print("Got frame")
while True:
    ret, _ = vid.read(frame)

    #cv2.imwrite("webcam_frame.png", frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
