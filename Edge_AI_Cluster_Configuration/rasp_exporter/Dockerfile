FROM python:3.9-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    bluetooth \
    bluez \
    libbluetooth-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir bleak prometheus_client

COPY rasp_power_exporter.py .

CMD ["python", "rasp_power_exporter.py"]
