#!/bin/bash
# Start the Yolov7 Docker container and the JSON-RPC server headless. For use with L4T r32.7.1.
docker run --restart unless-stopped --rm --runtime nvidia -e DISPLAY=$DISPLAY --network host -v /home/nvidia/data:/media/data -w /media/data/yolo-jetson r32.7.2-pth1.10-yolo python3 run_server_sync.py
