FROM ubuntu:22.04

# 작업 디렉토리 설정
WORKDIR /home
ENV HOME /home
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

VOLUME /data
EXPOSE 88

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git nano python3-pip python3-dev pkg-config wget usbutils vim curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip, numpy 설치 (numpy 2 미만)
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install --no-cache-dir "numpy<2"

# Coral EdgeTPU 런타임 저장소/키 추가 및 설치
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update \
    && apt-get install -y libedgetpu1-std \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# PyCoral, tflite_runtime 설치 (aarch64용 wheel)
RUN wget https://github.com/oberluz/pycoral/releases/download/2.13.0/pycoral-2.13.0-cp310-cp310-linux_aarch64.whl \
    && wget https://github.com/oberluz/pycoral/releases/download/2.13.0/tflite_runtime-2.13.0-cp310-cp310-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir pycoral-2.13.0-cp310-cp310-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir tflite_runtime-2.13.0-cp310-cp310-linux_aarch64.whl \
    && rm pycoral-2.13.0-cp310-cp310-linux_aarch64.whl \
    && rm tflite_runtime-2.13.0-cp310-cp310-linux_aarch64.whl

# FastAPI/uvicorn 등 서버용 패키지(추가 필수)
RUN python3 -m pip install --no-cache-dir fastapi uvicorn

# 코드/모델 복사
COPY . /home

# 기본 실행 포트/커맨드(예시: fastapi 서버)
EXPOSE 8080
CMD ["uvicorn", "coral_inference_server:app", "--host", "0.0.0.0", "--port", "8080"]

