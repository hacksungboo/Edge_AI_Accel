import os
import time
import base64
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
from datetime import datetime
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

MODEL_DICT = {
    "mobilenet": "/app/model/mobilenetv2_int8.engine",
    "resnet50": "/app/model/resnet50_int8.engine",
    "efficientnet": "/app/model/efficientnet_b1_int8.engine",
    "inception": "/app/model/inception-v1-12-int8.engine"
}
inference_lock = threading.Lock()
app = FastAPI(title="Jetson TensorRT Python API Inference Server")
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class InferRequest(BaseModel):
    model: str = "mobilenet"
    input: str  # base64 인코딩 이미지

class InferResponse(BaseModel):
    server: str
    start_time: str
    end_time: str
    inference_time: float
    result: str

def format_timestamp(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def get_server_name():
    server_name = os.uname().nodename
    server_name = os.environ.get("MY_NODE_NAME")
    if not server_name:
        server_name = os.uname().nodename
    return server_name

def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({"host": host_mem, "device": device_mem})
        else:
            outputs.append({"host": host_mem, "device": device_mem})
    return inputs, outputs, bindings, stream

@app.on_event("startup")
def startup_event():
    global engines, contexts, buffers
    engines = {}
    contexts = {}
    buffers = {}
    for model_name, model_path in MODEL_DICT.items():
        engine = load_engine(model_path)
        context = engine.create_execution_context()
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        engines[model_name] = engine
        contexts[model_name] = context
        buffers[model_name] = (inputs, outputs, bindings, stream)
        print(f"{model_name} engine loaded and context created.")

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    model_name = request.model
    if model_name not in engines:
        raise HTTPException(400, detail=f"Model {model_name} not supported.")

    try:
        img_b64 = request.input
        img_bytes = base64.b64decode(img_b64)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape((224, 224, 3))
    except Exception as e:
        raise HTTPException(400, detail="이미지 decode 오류: " + str(e))

    with inference_lock:
        try:
            start = time.time()
            inputs, outputs, bindings, stream = buffers[model_name]
            np.copyto(inputs[0]["host"], img_np.ravel())
            cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
            contexts[model_name].execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
            stream.synchronize()
            end = time.time()
            inference_time = round(end - start, 4)
            return InferResponse(
                server=get_server_name(),
                start_time=format_timestamp(start),
                end_time=format_timestamp(end),
                inference_time=inference_time,
                result="done"
            )
        except Exception:
            return InferResponse(
                server=get_server_name(),
                start_time=None,
                end_time=None,
                inference_time=None,
                result="error"
            )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("jetson_inference_server:app", host="0.0.0.0", port=8080)
