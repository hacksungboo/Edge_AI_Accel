import os
import time
import base64
import threading
from typing import Any, Dict, Optional

import numpy as np
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import hailo_platform as hpf

MODEL_DICT = {
    "mobilenet": "./model/mobilenet_v2_1.0.hef",
    "resnet50": "./model/resnet_v1_50.hef",
    "efficientnet": "./model/efficientnet_s.hef",
    "inception": "./model/inception_v1.hef"
    }

MODEL_CACHE: Dict[str, Dict[str, Any]] = {}

try:
    _global_device = hpf.VDevice()
except Exception as e:
    print(f"[ERROR] Failed to create global Hailo device: {e}")

inference_lock = threading.Lock()

app = FastAPI(title="Hailo Inference Server (With Hostname)")


def format_timestamp(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # 밀리초까지


def get_server_name():
    server_name = os.environ.get("MY_NODE_NAME")
    if not server_name:
        server_name = os.uname().nodename
    return server_name


class InferRequest(BaseModel):
    model: str  # MODEL_DICT 키
    input: str  # base64 인코딩된 raw uint8 tensor (shape: (1, *input_shape))


class InferResponse(BaseModel):
    server: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    inference_time: Optional[float] = None  # seconds
    result: Any = None  # 출력 요약 또는 에러


def load_models():
    for name, path in MODEL_DICT.items():
        if not os.path.exists(path):
            print(f"[WARN] HEF {path} for model '{name}' not found. Skipping.")
            continue
        try:
            hef = hpf.HEF(path)
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef, interface=hpf.HailoStreamInterface.PCIe
            )
            network_group = _global_device.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()

            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_info = hef.get_output_vstream_infos()[0]

            input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                network_group, quantized=True, format_type=hpf.FormatType.AUTO
            )
            output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                network_group, quantized=True, format_type=hpf.FormatType.AUTO
            )

            MODEL_CACHE[name] = {
                "hef": hef,
                "network_group": network_group,
                "network_group_params": network_group_params,
                "input_info": input_vstream_info,
                "output_info": output_vstream_info,
                "input_vstreams_params": input_vstreams_params,
                "output_vstreams_params": output_vstreams_params,
            }

            print(f"Loaded Hailo model '{name}' from {path}")
            print(f"  Input shape: {input_vstream_info.shape}, Output shape: {output_vstream_info.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load model '{name}' from {path}: {e}")


load_models()


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest):
    model_name = request.model
    if model_name not in MODEL_CACHE:
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' not loaded or missing.")

    entry = MODEL_CACHE[model_name]
    network_group = entry["network_group"]
    network_group_params = entry["network_group_params"]
    input_info = entry["input_info"]
    output_info = entry["output_info"]
    input_vstreams_params = entry["input_vstreams_params"]
    output_vstreams_params = entry["output_vstreams_params"]

    try:
        raw = base64.b64decode(request.input)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 디코딩 실패: {e}")

    expected_shape = (1, *input_info.shape)
    expected_bytes = int(np.prod(expected_shape))
    if len(raw) != expected_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"입력 크기 불일치: 모델 '{model_name}'이 기대하는 바이트 수는 {expected_bytes}이지만 "
                f"받은 길이는 {len(raw)}입니다. 예상 shape: {expected_shape}, dtype: uint8."
            ),
        )
    try:
        input_array = np.frombuffer(raw, dtype=np.uint8).copy().reshape(expected_shape)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"입력 배열 변환 실패: {e}")

    with inference_lock:
        try:
            start = time.time()
            with network_group.activate(network_group_params):
                with hpf.InferVStreams(
                    network_group,
                    input_vstreams_params,
                    output_vstreams_params,
                ) as infer_pipeline:
                    input_data = {input_info.name: input_array}
                    results = infer_pipeline.infer(input_data)
            end = time.time()

            inference_time = round(end - start, 4)
            output_tensor = results.get(output_info.name)
            result_summary: Dict[str, Any] = {}

            if output_tensor is not None:
                result_summary["output_shape"] = output_tensor.shape
                result_summary["dtype"] = str(output_tensor.dtype)
            else:
                result_summary["note"] = f"출력키 '{output_info.name}' 없음."

            return InferResponse(
                server=get_server_name(),
                start_time=format_timestamp(start),
                end_time=format_timestamp(end),
                inference_time=inference_time,
                result="done"
            )
        except Exception as e:
            err_msg = str(e)
            print(f"[ERROR] Inference failed for model '{model_name}': {err_msg}")
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

    uvicorn.run("hailo_inference_server:app", host="0.0.0.0", port=8080)

