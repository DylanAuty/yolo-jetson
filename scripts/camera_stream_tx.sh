#!/bin/bash
# Begin streaming existing video stream to target IP.

ip=172.22.83.251
port=5000
if [ $# -gt 0 ]
then
    ip=$1
fi

echo Streaming to $ip on port $port
gst-launch-1.0 -v v4l2src device=/dev/video0 ! jpegdec ! video/x-raw,framerate=12/1 ! x264enc tune=zerolatency ! rtph264pay ! udpsink host=$ip port=$port
