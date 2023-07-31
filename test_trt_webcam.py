import cv2
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import random
import threading
import time

import yolojetson.utils
import yolojetson.constants
from yolojetson.VideoCaptureThreading import VideoCaptureThreading

class BaseEngine(object):
    def __init__(self, engine_path, imgsz=(640,640)):
        self.imgsz = imgsz
        self.mean = None
        self.std = None
        self.class_names = [ 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger,'')
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
                
    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        # transfer data to the gpu
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        # run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # fetch outputs from gpu
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        # synchronize stream
        self.stream.synchronize()

        data = [out['host'] for out in self.outputs]
        return data

    def inference(self, img_path, conf=0.25):
        origin_img = cv2.imread(img_path)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        img, ratio = self.preproc(origin_img, self.imgsz, self.mean, self.std)
        
        num, final_boxes, final_scores, final_cls_inds = self.infer(img)
        final_boxes = np.reshape(final_boxes, (-1, 4))
        num = num[0]

        if num >0:
            final_boxes, final_scores, final_cls_inds = final_boxes[:num]/ratio, 1+final_scores[:num], final_cls_inds[:num]
            origin_img = yolojetson.utils.vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)                      
        return origin_img

    def inference_image(self, origin_img, conf=0.25):
        #origin_img = cv2.imread(img_path)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2RGB)
        img, ratio = self.preproc(origin_img, self.imgsz, self.mean, self.std)
        
        num, final_boxes, final_scores, final_cls_inds = self.infer(img)
        final_boxes = np.reshape(final_boxes, (-1, 4))
        num = num[0]

        if num >0:
            final_boxes, final_scores, final_cls_inds = final_boxes[:num]/ratio, 1+final_scores[:num], final_cls_inds[:num]
            origin_img = yolojetson.utils.vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2BGR)                      
        return origin_img

    def get_fps(self):
        # warmup
        import time
        img = np.ones((1,3,self.imgsz[0], self.imgsz[1]))
        img = np.ascontiguousarray(img, dtype=np.float32)
        for _ in range(20):
            _ = self.infer(img)
        t1 = time.perf_counter()
        _ = self.infer(img)
        print(1/(time.perf_counter() - t1), 'FPS')


    def preproc(self, image, input_size, mean, std, swap=(2, 0, 1)):
        if len(image.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
        else:
            padded_img = np.ones(input_size) * 114.0
        img = np.array(image)
        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img[:, :, ::-1]
        padded_img /= 255.0
        if mean is not None:
            padded_img -= mean
        if std is not None:
            padded_img /= std
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r




def main():
    print("Setting up video stream")
    video = VideoCaptureThreading('\
            udpsrc port=5000 \
            ! application/x-rtp,encoding-name=H264,payload=96 \
            ! rtph264depay \
            ! avdec_h264 \
            ! videoconvert \
            ! video/x-raw,format=BGR \
            ! appsink sync=false drop=true \
            ', cv2.CAP_GSTREAMER)
    print("Setting up prediction engine")
    pred = BaseEngine(engine_path='./checkpoints/yolov7_640-nms.trt', imgsz=(360,640))
    print("Starting capture loop")
    video.start()
    start_time = None
    while True:
        start_time = time.time()
        ret, image = video.read()
        origin_img = pred.inference_image(image)
        cv2.imshow('frame', origin_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.setWindowTitle('frame', f'FPS: {1 / (time.time() - start_time):4.2}')
    pred.get_fps()
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
