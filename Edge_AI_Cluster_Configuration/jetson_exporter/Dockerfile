FROM python:3.9-slim
  
WORKDIR /app

RUN apt-get update && \
    apt-get install -y libatlas-base-dev && \
    pip install jetson-stats prometheus_client

COPY custom_exporter.py .

CMD ["python", "custom_exporter.py"]

