#!/bin/bash
# Start the Yolov7 Docker container and the JSON-RPC server in the background. For use with L4T r32.7.1.
docker run -d --restart unless-stopped --runtime nvidia --network host -v /home/$USER/Documents/yolo-jetson:/media/yolo-jetson -w /media/yolo-jetson yolo-jetson:latest bash -c "python3 -m pip install jsonrpclib-pelix && python3 run_server_sync.py > logs/server_log.txt 2>&1"
