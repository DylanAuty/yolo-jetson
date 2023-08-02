# run_server_sync.py
# Instantiate and start the JSON-RPC YOLO inference server.
# This version only runs inference on-demand, and so is slower.
import argparse
import time
import cv2
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer

import yolojetson.utils
from yolojetson.VideoCaptureThreading import VideoCaptureThreading
from yolojetson.TRTBaseEngine import TRTBaseEngine

class ServerAsync:
    def __init__(self, args):
        self.args = args
        self.video = VideoCaptureThreading(f'\
                 udpsrc port={args.video_port} \
                 ! application/x-rtp,encoding-name=H264,payload=96 \
                 ! rtph264depay \
                 ! avdec_h264 \
                 ! videoconvert \
                 ! video/x-raw,format=BGR \
                 ! appsink sync=false drop=true \
                 ', cv2.CAP_GSTREAMER)
        self.engine = TRTBaseEngine(engine_path=args.checkpoint, imgsz=(640,640))


    def start(self):
        # Start video stream and server
        self.video.start()
        self.server = SimpleJSONRPCServer(('0.0.0.0', self.args.port))
        self.server.register_function(self.run_inference)
        self.server.serve_forever()
        print(f'Server listening on port {self.args.port}...')


    def run_inference(self):
        ret, image = self.video.read()
        origin_img, most_recent_results = self.engine.inference_image(image)
        most_recent_results['timestamp'] = time.time()
        detections_json = yolojetson.utils.detection_to_json(most_recent_results, class_names=self.engine.class_names)
        return detections_json


def main(args):
    # Instantiate and start up the server
    server = ServerAsync(args)
    server.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default='./checkpoints/yolov7_640-nms.trt', help='Path to the tensorrt checkpoint/engine to use.')
    parser.add_argument('--video_port', type=int, default=5000, help='Which port to listen on for incoming video')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Which port to serve JSON-RPC requests on.')

    args = parser.parse_args()
    main(args)
