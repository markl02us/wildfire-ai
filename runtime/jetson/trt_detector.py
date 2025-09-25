import cv2, numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

class TRTDetector:
    def __init__(self, engine_path, conf=0.35, input_size=(640,640)):
        self.conf = conf
        self.input_size = input_size
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    def allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            device_mem = cuda.mem_alloc(size * np.dtype(dtype).itemsize)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inputs.append({"host": cuda.pagelocked_empty(size, dtype), "device": device_mem})
            else:
                outputs.append({"host": cuda.pagelocked_empty(size, dtype), "device": device_mem})
        return inputs, outputs, bindings, stream

    def preprocess(self, frame):
        img = cv2.resize(frame, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2,0,1))[None]
        return img

    def __call__(self, frame):
        img = self.preprocess(frame)
        np.copyto(self.inputs[0]["host"], img.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for o in self.outputs:
            cuda.memcpy_dtoh_async(o["host"], o["device"], self.stream)
        self.stream.synchronize()
        preds = np.array(self.outputs[0]["host"]).reshape(-1, 6)
        detections = [p for p in preds if p[4] >= self.conf]
        return detections
