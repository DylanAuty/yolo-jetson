#!/bin/bash
# Convenience script to start the Yolov7 Docker container for use with L4T r32.7.2.
docker run -it --rm --runtime nvidia -e DISPLAY=$DISPLAY --network host -v /home/$USER/Documents/yolo-jetson:/media/yolo-jetson -w /media/yolo-jetson yolo-jetson:latest
