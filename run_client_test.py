# run_client_test.py
# Instantiates a JSON-RPC client, then polls it for results as fast as it can.
import argparse
import time
import jsonrpclib
import json

def main(args):
    server = jsonrpclib.ServerProxy(args.address)
    while True:
        start_time = time.time()
        if args.visualise:
            json_response, annotated_image = server.run_inference(conf=0.25, do_visualise=True)
            cv2.imshow('frame', annotated_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.setWindowTitle('frame', f'FPS: {1 / (time.time() - start_time):4.2f}')
            print(json_response)
        else:
            json_response = server.run_inference()
            print(json_response)
            print(f'FPS: {1 / (time.time() - start_time):4.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', '-a', default='http://127.0.0.1:8080', help='Address and port of server.')
    parser.add_argument('--visualise', '-v', action='store_true', help='Whether to have the server return annotated frames.')

    args = parser.parse_args()
    main(args)
