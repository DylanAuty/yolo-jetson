#!/bin/bash
# Start the Yolov7 Docker container and the JSON-RPC server in the background. For use with L4T r32.7.1.
ver_maj=$(cat /etc/nv_tegra_release | cut -f2 -d ' ' | sed 's/R/r/g')
ver_min=$(cat /etc/nv_tegra_release | cut -f5 -d ' ' | sed 's/,//g')

docker run -d --restart unless-stopped --runtime nvidia --network host -v /home/$USER/Documents/yolo-jetson:/media/yolo-jetson -w /media/yolo-jetson yolo-jetson:$ver_maj.$ver_min bash -c "python3 run_server_sync.py > logs/server_log.txt 2>&1"
