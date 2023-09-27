#!/bin/bash
# Set up the jetson container.

cd jetson-containers/
./build.sh --name=yolo-jetson opencv_cuda pytorch pycuda python-jsonrpclib-pelix python-pyequilib python-ultralytics
cd ..
