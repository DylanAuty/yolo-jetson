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
        response = server.run_inference()
        #print(response)
        print(f'FPS: {1 / (time.time() - start_time):4.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', '-a', default='http://127.0.0.1:8080', help='Address and port of server.')

    args = parser.parse_args()
    main(args)
