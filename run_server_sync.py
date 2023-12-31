# run_server_sync.py
# Instantiate and start the JSON-RPC YOLO inference server.
# This version only runs inference on-demand, and so is slower.
import argparse
import os
import time
import cv2
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer

import yolojetson.utils
from yolojetson.VideoCaptureThreading import VideoCaptureThreading
from yolojetson.TRTBaseEngine import TRTBaseEngine

class ServerSync:
    def __init__(self, args):
        self.args = args
        if self.args.save_video:
            self.video_save_dir = os.path.join("saved_runs", f"capture_{time.strftime('%Y-%m-%d_%H-%M-%S')}")
            if not os.path.exists(self.video_save_dir):
                os.makedirs(self.video_save_dir)
            print(f"Saving video capture to {self.video_save_dir}")
        self.video = VideoCaptureThreading(f'\
                 udpsrc address={self.args.video_ip} port={self.args.video_port} \
                 ! application/x-rtp,clock-rate=90000,encoding-name=H264,payload=96 \
                 ! rtpjitterbuffer latency=1000 \
                 ! rtph264depay \
                 ! queue \
                 ! avdec_h264 \
                 ! videoconvert \
                 ! video/x-raw,format=BGR \
                 ! appsink sync=false \
                 ', cv2.CAP_GSTREAMER)
        if "yolov8" in self.args.checkpoint:
            from ultralytics import YOLO
            self.engine = YOLO(args.checkpoint)
        else:
            self.engine = TRTBaseEngine(engine_path=self.args.checkpoint, imgsz=(self.args.resolution[0], self.args.resolution[1]))


    def start(self):
        # Start video stream and server
        self.video.start()
        self.server = SimpleJSONRPCServer((self.args.server_ip, self.args.port))
        self.server.register_function(self.run_inference)
        self.server.serve_forever()
        print(f'Server listening on port {self.args.port}...')


    def run_inference(self):
        ret, image = self.video.read()
        timestamp = time.time()
        if "yolov8" in args.checkpoint:
            results = self.engine(image, imgsz=self.args.resolution)
            if self.args.save_video:
                annotated_img = results[0].plot()
            else:
                annotated_img = None
            most_recent_results = results[0].tojson()
            detections_json = most_recent_results
        else:
            annotated_img, most_recent_results = self.engine.inference_image(image, do_visualise=self.args.save_video)
            most_recent_results['timestamp'] = timestamp
            detections_json = yolojetson.utils.detection_to_json(most_recent_results, class_names=self.engine.class_names)

        if self.args.save_video:
            filename = os.path.join(self.video_save_dir, f"{str(timestamp)}.png")
            cv2.imwrite(filename, annotated_img)

        return detections_json


def main(args):
    # Instantiate and start up the server
    server = ServerSync(args)
    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, default='./checkpoints/yolov7_640-nms.trt', help='Path to the tensorrt checkpoint/engine to use.')
    parser.add_argument('--resolution', '-r', type=str, default="640,640", help='Resolution of video as a comma separated list (e.g. "width,height"). Should normally be square (640,640 or 1280,1280). Should match checkpoint (e.g. yolov8n-1280 needs 1280x1280 input).')
    parser.add_argument('--video_port', type=int, default=5000, help='Which port to listen on for incoming video')
    parser.add_argument('--video_ip', type=str, default="0.0.0.0", help='Which of this device\'s ips to listen on for incoming video. Necessary for devices with multiple ips on a single interface.')
    parser.add_argument('--server_ip', type=str, default="0.0.0.0", help='Which of this device\'s ips to serve on. Necessary for devices with multiple ips on a single interface.')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Which port to serve JSON-RPC requests on.')
    parser.add_argument('--save_video', action='store_true', help='Save the annotated video frames to a new output directory in saved_runs')

    args = parser.parse_args()
    args.resolution = [int(item) for item in args.resolution.split(',')]
    main(args)
