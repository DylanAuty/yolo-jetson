# run_server_async.py
# Instantiate and start the JSON-RPC YOLO inference server.
# This version runs inference constantly, so requests should be served instantly.
import argparse
import cv2
from jsonrpclib.SimpleJSONRPCServer import SimpleJSONRPCServer

from yolojetson.InferenceEngineThreaded import InferenceEngineThreaded

def main(args):
    engine = InferenceEngineThreaded(
                    video_src=f'udpsrc port={args.video_port} \
                              ! application/x-rtp,encoding-name=H264,payload=96 \
                              ! rtph264depay \
                              ! avdec_h264 \
                              ! videoconvert \
                              ! video/x-raw,format=BGR \
                              ! appsink sync=false drop=true',
                    video_api_pref=cv2.CAP_GSTREAMER,
                    checkpoint=args.checkpoint)
    engine.start()
    while True:
        detections_json, annotated_image = engine.read()
        cv2.imshow('frame', annotated_image)
        print(detections_json)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', default='./checkpoints/yolov7_640-nms.trt', help='Path to the tensorrt checkpoint/engine to use.')
    parser.add_argument('--video_port', type=int, default=5000, help='Which port to listen on for incoming video')
    parser.add_argument('--port', '-p', type=int, default=8080, help='Which port to serve JSON-RPC requests on.')

    args = parser.parse_args()
    main(args)
