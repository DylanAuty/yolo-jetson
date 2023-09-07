#!/bin/bash
# Receive video stream encoded as h264 over UDP, and try to display it.

port=5000
if [ $# -gt 0 ]
then
    port=$1
fi

echo Receiving stream on port $port
# For h264 - sends partial frames and can lead to corrupted images on receiving end
gst-launch-1.0 -v udpsrc port=$port ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false

# For JPEG - no corruption visible
#gst-launch-1.0 -v udpsrc port=5000 ! application/x-rtp,encoding-name=JPEG,payload=96 ! rtpjpegdepay ! jpegdec ! videoconvert ! ximagesink sync=false
