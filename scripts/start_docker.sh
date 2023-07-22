#!/bin/bash
# Convenience script to start the Yolov7 Docker container for use with L4T r32.7.1.
docker run -it --rm --runtime nvidia -e DISPLAY=$DISPLAY --network host -v /home/nvidia/data:/media/data -w /media/data r32.7.2-pth1.10-yolo 
