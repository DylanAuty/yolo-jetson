#!/bin/bash
# Begin streaming existing video stream to target IP.

ip=172.22.83.251
port=5000
if [ $# -gt 0 ]
then
    ip=$1
fi

echo Streaming to $ip on port $port
# For converting MJPEG camera output to H264. Can result in dropped/partial frames on receiving end (looks like washed out grey video)
gst-launch-1.0 -v v4l2src device=/dev/video0 ! jpegdec ! video/x-raw,frame-rate=30/1 ! x264enc tune=zerolatency ! rtph264pay ! application/x-rtp,encoding-name=H264,payload=96 ! udpsink host=$ip port=$port

# For sending MJPEG camera stream directly.
#gst-launch-1.0 -v v4l2src device=/dev/video0 ! image/jpeg,frame-rate=30/1 ! rtpjpegpay ! udpsink host=$ip port=$port
