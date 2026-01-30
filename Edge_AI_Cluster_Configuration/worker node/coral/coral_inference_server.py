import os
import time
import numpy as np
import base64
import threading
from typing import Any
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

MODEL_DICT = {
    "mobilenet": "./model/mobilenet_v2_1.0_224_quant_edgetpu.tflite",
    "resnet50": "./model/tfhub_tf2_resnet_50_imagenet_ptq_edgetpu.tflite",
    "efficientnet": "./model/efficientnet-edgetpu-S_quant_edgetpu.tflite",
    "inception": "./model/inception_v1_224_quant_edgetpu.tflite"
}
INTERPRETER_CACHE = {}
for m_name, m_path in MODEL_DICT.items():
    if os.path.exists(m_path):
        itp = make_interpreter(m_path)
        itp.allocate_tensors()
        INTERPRETER_CACHE[m_name] = itp
        print(f"{m_name} loaded from {m_path}")
    else:
        print(f"{m_name} not found. Skip.")

inference_lock = threading.Lock()

app = FastAPI(title="Coral Inference Server (With Hostname)")

def format_timestamp(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 밀리초까지

def get_server_name():
    # 환경변수 우선, 없으면 uname
    server_name = os.environ.get("MY_NODE_NAME")
    if not server_name:
        server_name = os.uname().nodename
    return server_name

class InferRequest(BaseModel):
    model: str = "mobilenet"
    input: str  # base64 인코딩 이미지

class InferResponse(BaseModel):
    server: str
    start_time: str = None
    end_time: str = None
    inference_time: float = None
    result: str = None

@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    model_name = request.model
    img_b64 = request.input
    if model_name not in INTERPRETER_CACHE:
        raise HTTPException(400, detail=f"Model {model_name} not loaded.")
    try:
        img_bytes = base64.b64decode(img_b64)
        img_np = np.frombuffer(img_bytes, dtype=np.uint8).reshape((224, 224, 3))
    except Exception as e:
        raise HTTPException(400, detail="이미지 decode 오류: "+str(e))
    interpreter = INTERPRETER_CACHE[model_name]
    with inference_lock:
        try:
            start = time.time()
            common.set_input(interpreter, img_np)
            interpreter.invoke()
            end = time.time()
            inference_time = round(end - start, 4)  # 적당히 반올림(초 단위)
            return InferResponse(
                server=get_server_name(),
                start_time=format_timestamp(start),
                end_time=format_timestamp(end),
                inference_time=inference_time,
                result="done"
            )
        except Exception as e:
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
    uvicorn.run("coral_inference_server:app", host="0.0.0.0", port=8080)
