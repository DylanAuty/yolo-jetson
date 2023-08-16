#!/bin/bash
# Convenience script to start the Yolov7 Docker container.
ver_maj=$(cat /etc/nv_tegra_release | cut -f2 -d ' ' | sed 's/R/r/g')
ver_min=$(cat /etc/nv_tegra_release | cut -f5 -d ' ' | sed 's/,//g')

docker run -it --rm --runtime nvidia -e DISPLAY=$DISPLAY --network host -v /home/$USER/Documents/yolo-jetson:/media/yolo-jetson -w /media/yolo-jetson yolo-jetson:$ver_maj.$ver_min
