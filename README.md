# YOLO-Jetson

This repository contains scripts for working with YOLO on an NVIDIA Jetson. Video is received over ethernet, and results are served back over the network as JSON.

The work here builds on the work done by Adrian Lopez-Rodriguez and Nelson Da Silva at Imperial College London.


## Quickstart TL;DR (detailed instructions below)
1. Install Docker and enable non-root docker management
2. Install container dependencies: `python3 -m pip install -r jetson-containers/requirements.txt" 
3. Add `"default-runtime": "nvidia"` to your `/etc/docker/daemon.json`
4. Build container: `./scripts/setup_container.sh`
5. (On camera machine): Start webcam stream: `./scripts/camera_stream_tx.sh <server_ip>`
6. (On Jetson) Start server headless: `./scripts/start_server_headless.sh`
	- _Optional:_ view logs with `tail -f ./logs/server_log.txt`
	- _To start from shell within docker container instead:_: `./scripts/start_docker.sh`, then `python3 run_server_sync.py`
7. (On client) Start test client: `python3 -m pip install argparse json jsonrpclib-pelix && python3 run_client_test.py http://<server_ip>:8080`
	- The `http://` and port number are necessary.

Server tested on Python 3.8.10 running on a Jetson Orin NX running L4T r35.3.1. Client tested on Python 3.10.8.

You also need checkpoints in `.trt` format. You can download a test version from [this Github release](https://github.com/DylanAuty/yolo-jetson/releases/download/v0.2/yolov7_640-nms.trt). Place it in the `./checkpoints` directory. Note that this checkpoint has not been tuned on fisheye video and is intended for testing the system.

## Files
- `run_client_test.py`: Start the test client. Run with `-h` to see options.
- `run_server_sync.py`: To run within docker image. Starts the example server that runs inference on demand and returns a JSON with the detections.
- `run_server_async.py`: **Not currently working** but provided in case it's useful, this is meant to be a threaded version that runs inference constantly in an attempt to improve FPS at the expense of power consumption.
- `scripts/`:
	- `setup_container.sh`: Convenience script to setup the jetson container.
	- `camera_stream_tx.sh <target ip>`: Begin streaming h264-encoded video using RTP over UDP to the target IP on port 5000. 
	- `camera_stream_rx.sh <listen port>`: Test to receive h264-encoded video from an RTP-over-UDP source. Also includes commented line for JPEG-encoded video. Doesn't always work due to varying availability of display sinks in different environments.
	- `start_docker.sh`: Start the working docker image in interactive mode.
	- `start_server_headless.sh`: Start the server within its docker image and have it run in the background. Log output is redirected to `logs/server_log.txt`.
	- `test_python_webcam.py`: Python script to test GStreamer + CV2 pipelines within python.
- `jetsoncontainers/`: A fork of the [official jetson-containers repo](https://github.com/dusty-nv/jetson-containers) from NVIDIA.
	- Previously this fork contained modifications to container dockerfiles and build scripts, but the upstream repo has since had a major overhaul and now works with minimal modification. Only modification now is a custom package called `python-jsonrpclib-pelix`.

## Environment setup
The application runs inside a Docker image, using the nvidia-container-runtime to give access to the GPU.

### Docker installation and setup
First, install Docker according to the distro being used. Then, [set up docker for management as a non-root user](https://docs.docker.com/engine/install/linux-postinstall/):
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
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
Then reboot the system before continuing.

## Building Docker container
Due to changes to the upstream repo, this is now much simpler: just run `./scripts/setup_container.sh`. This installs all the required dependencies, and will also build OpenCV from source to ensure that GStreamer support is included.

## Running the server
To start the server from within the Docker container, start the Docker container and then run `python3 run_server_sync.py`. By default, this will serve over HTTP on port 8080. This can be changed if needed.

The server and container can be started from outside the Docker container by running `./scripts/start_server_headless.sh`.

## Running a client
An example test client that repeatedly polls the server for detections is found in `run_client_test.py`. The server exposes one method: `run_inference()` will grab a video frame, run inference on it, and return a JSON string containing the detections. 

## Detection JSON Format
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

