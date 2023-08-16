#!/bin/bash
# Set up the jetson container.

cd jetson-containers/
./build.sh --name=yolo-jetson:latest opencv_cuda pytorch pycuda python-jsonrpclib-pelix
cd ..
