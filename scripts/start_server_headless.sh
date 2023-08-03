#!/bin/bash
# Start the Yolov7 Docker container and the JSON-RPC server in the background. For use with L4T r32.7.1.
docker run -d --restart unless-stopped --runtime nvidia --network host -v /home/nvidia/data:/media/data -w /media/data/yolo-jetson r32.7.2-pth1.10-yolo bash -c "python3 run_server_sync.py > logs/server_log.txt 2>&1"
