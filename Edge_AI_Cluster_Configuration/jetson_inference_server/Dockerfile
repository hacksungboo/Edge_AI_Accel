FROM nvcr.io/nvidia/l4t-tensorflow:r32.7.1-tf2.7-py3

WORKDIR /home
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y \
    python3-pip python3-dev python3-setuptools \
    nano wget curl git vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir "numpy<2"

# uvicorn[standard] 로 설치 (권장)
RUN python3 -m pip install fastapi uvicorn pycuda

COPY . /home

EXPOSE 8080

CMD ["uvicorn", "jetson_inference_server:app", "--host", "0.0.0.0", "--port", "8080"]

