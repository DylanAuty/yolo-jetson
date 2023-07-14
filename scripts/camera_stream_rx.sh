#!/bin/bash
# Receive video stream encoded as h264 over UDP, and try to display it.

port=5000
if [ $# -gt 0 ]
then
    port=$1
fi

echo Receiving stream on port $port
gst-launch-1.0 -v udpsrc port=$port ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! ximagesink sync=false
