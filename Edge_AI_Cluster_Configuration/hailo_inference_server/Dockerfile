FROM qngkrtjd/hailort_img:arm64

WORKDIR /app

RUN pip3 install --no-cache-dir fastapi uvicorn

COPY hailo_inference_server.py /app/

EXPOSE 8080

CMD ["uvicorn", "hailo_inference_server:app", "--host", "0.0.0.0", "--port", "8080"]

