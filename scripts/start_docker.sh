#!/bin/bash
# Convenience script to start the Yolov7 Docker container for use with L4T r32.7.1.
# fbf5bc2 is the hash for the relevant L4T image with CUDA, Pytorch, and the Yolov7 dependencies.
# Repository: nvcr.io/nvidia/l4t-pytorch
# Tag: r32.7.1-pth1.10-py3-yolo
docker run -it --rm --runtime nvidia -e DISPLAY=$DISPLAY --network host -v /home/nvidia/data:/media/data -w /media/data fbf5bc2
