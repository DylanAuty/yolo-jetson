# run_server_sync.py
# Instantiate and start the JSON-RPC YOLO inference server.
# This version only runs inference on-demand, and so is slower.
import argparse
import cv2
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer

def main(args):
    # Setup stream and inference engine
    global video = VideoCaptureThreading(f'\
             udpsrc port={args.video_port} \
             ! application/x-rtp,encoding-name=H264,payload=96 \
             ! rtph264depay \
             ! avdec_h264 \
             ! videoconvert \
             ! video/x-raw,format=BGR \
             ! appsink sync=false drop=true \
             ', cv2.CAP_GSTREAMER)
    global engine = TRTBaseEngine(engine_path=args.checkpoint, imgsz=(640,640))
    video.start()

    # Setup server and start
    server = SimpleJSONRPCServer(('127.0.0.1', args.port))
    server.register_function(run_inference)
    server.serve_forever()


def run_inference():
    detections_json, annotated_image = engine.read()
    print(detections_json)
    return detections_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default='./checkpoints/yolov7_640-nms.trt', help='Path to the tensorrt checkpoint/engine to use.')
    parser.add_argument('--video_port', type=int, default=5000, help='Which port to listen on for incoming video')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Which port to serve JSON-RPC requests on.')

    args = parser.parse_args()
    main(args)
