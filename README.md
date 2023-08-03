# YOLO-Jetson

This repository contains scripts for working with YOLO on an NVIDIA Jetson. Video is received over ethernet, and results are served back over the network as JSON.

The work here builds on the work done by Adrian Lopez-Rodriguez and Nelson Da Silva at Imperial College London.


## Quickstart TL;DR
1. Install Docker and enable non-root docker management
2. Add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json`:
3. Build container: `cd jetson-containers && ./scripts/docker_build_yolo.sh`.
	- This step will probably fail at first on any non-TX2 device. [See container setup section](### Building Docker container).
4. (On Jetson) Start server: `./scripts/start_server_headless.sh`
	- _Optional:_ view logs with `tail -f ./logs/server_log.txt`
5. (On client) Start test client: `python3 -m pip install argparse json jsonrpclib-pelix && python3 run_client_test.py`

Server tested on Python 3.6.9 running on a Jetson TX2 running L4T r32.7.2. Client tested on Python 3.10.8 on the client.


## Files
- `run_client_test.py`: Start the test client. Run with `-h` to see options.
- `run_server_sync.py`: To run within docker image. Starts the example server that runs inference on demand and returns a JSON with the detections.
- `run_server_async.py`: Not currently working, is meant to be a threaded version that runs inference constantly in an attempt to improve FPS at the expense of power consumption.
- `scripts/`:
	- `camera_stream_tx.sh <target ip>`: Begin streaming h264-encoded video using RTP over UDP to the target IP on port 5000. 
	- `camera_stream_rx.sh <listen port>`: Test to receive h264-encoded video from an RTP-over-UDP source. Also includes commented line for JPEG-encoded video. Doesn't always work due to varying availability of display sinks in different environments.
	- `start_docker.sh`: Start the working docker image in interactive mode.
	- `start_server_headless.sh`: Start the server within its docker image and have it run in the background. Log output is redirected to `logs/server_log.txt`.
	- `test_python_webcam.py`: Python script to test GStreamer + CV2 pipelines within python.
- `jetsoncontainers/`: A fork of the [official jetson-containers repo](https://github.com/dusty-nv/jetson-containers) from NVIDIA.
	- _Various Dockerfiles_: Not to be used directly. The only two used for this are `Dockerfile.opencv` and `Dockerfile.yolo`.
	- `scripts/docker_build_yolo.sh`: Will first build OpenCV from source (necessary for GStreamer support), then will build the server image and install dependencies. See below for usage instructions.


## Environment setup on the Jetson TX2
The application runs inside a Docker image, using the nvidia-container-runtime to give access to the GPU. The docker image is based on the pre-made Jetson docker containers available from [ the jetson-containers repo ](https://github.com/dusty-nv/jetson-containers), but the dockerfile has been modified to include OpenCV compiled from source with GStreamer support.

### Docker installation and setup
First, install Docker according to the distro being used. Then, [set up docker for management as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/):
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

Modify the default docker runtime by adding `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json`:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
```
Then reboot the system or restart the Docker daemon before continuing.

### Building Docker container
The version of OpenCV packaged with the off-the-shelf docker images doesn't have GStreamer support. The docker container build script will first compile OpenCV from source with the required support in a single-use container, then will build a fresh container to run the object detection models.

First modify `jetson-containers/scripts/docker_build_opencv.sh` so that `$cuda_arch_bin` matches the CUDA architecture for the target device. The Jetson TX2 architecture is SM62 (or compute_62), so this string should contain "6.2". Then, if needed, modify PYTORCH_VERSION inside `jetson-containers/scripts/docker_build_yolo.sh` to a version that works with the version of L4T flashed to the Jetson. This is used to find the correct starting image. [This page has a list of available images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch).

Then run:
```bash
cd jetson-containers
./scripts/docker_build_yolo.sh
```
which will compile OpenCV into .deb files, tar them together, and put them in `jetson-containers/packages`, before installing them in a new docker image. The new docker image will also have all required Python packages installed.

## Running the server
To start the server from within the Docker container, start the Docker container and then run `python3 run_server_sync.py`. By default, this will serve over HTTP on port 8080. This can be changed if needed.

The server and container can be started from outside the Docker container by running `./scripts/start_server_headless.sh`.

## Running a client
An example test client that repeatedly polls the server for detections is found in `run_client_test.py`. The server exposes one method: `run_inference()` will grab a video frame, run inference on it, and return a JSON string containing the detections. 

## JSON Format
The JSON containing the detections contains:
	- The unix timestamp field `timestamp` from the moment the video frame was grabbed,
	- N other fields representing N detections. Each detection's key in the JSON object is an integer beginning at 0. Detection objects contain:
		- `bbox`: Array of start and end coordinates for bounding box in the form (x, y, x, y), 
		- `conf`: Confidence of the detection, 
		- `class_idx`: Class index, 
		- `class_name`: Class name.

An example with two detections is shown below:

```json
{
	"timestamp": 1691000693.0368967,
    "0": {
        "bbox": [
            617.0,
            328.0,
            656.0,
            358.0
        ],
        "conf": "0.4790039",
        "class_idx": "9",
        "class_name": "motor"
    },
    "1": {
        "bbox": [
            574.0,
            365.5,
            621.0,
            401.0
        ],
        "conf": "0.44335938",
        "class_idx": "9",
        "class_name": "motor"
    }
}
```

