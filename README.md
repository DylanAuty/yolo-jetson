# YOLO-Jetson

This repository contains scripts for working with YOLO on an NVIDIA Jetson. Video is received over ethernet, and results are served back over the network as JSON.

The work here builds on the work done by Adrian Lopez-Rodriguez and Nelson Da Silva at Imperial College London.

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
./scripts/docker-build-yolo.sh
```
which will compile OpenCV into .deb files, tar them together, and put them in `jetson-containers/packages`, before installing them in a new docker image. The new docker image will also have all required Python packages installed.

## Running the server
To start the server from within the Docker container, start the Docker container and then run `python3 run_server_sync.py`. By default, this will serve over HTTP on port 8080. This can be changed if needed.

The server and container can be started from outside the Docker container by running `./scripts/start_server_headless.sh`.

## Running a client
An example test client that repeatedly polls the server for detections is found in `run_client_test.py`. The server exposes one method: `run_inference()` will grab a video frame, run inference on it, and return a JSON string containing the detections. If visualisations are required, then `run_inference(conf=0.5, return_visualisation=True)` can be used, and a tuple of `detections_json, annotated_image` will be returned from the server. `conf` is the confidence threshold in the range [0.0, 1.0] at which an annotation will be drawn onto the image. Image output is off by default for performance reasons.

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

