# Traffic Violation Detection System — container image
#
# Python 3.11 is required: paddlepaddle / paddleocr and torch publish wheels for
# 3.11 but NOT for 3.13, so building on 3.11 avoids the dependency dead-end you
# hit on a bare 3.13 install.
#
# This is a CPU image so it runs anywhere (laptops, cloud, judges' machines).
# For GPU, swap the base for `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`,
# install python3.11, and set device: "cuda" in configs/pipeline.yaml.

FROM python:3.11-slim

# OpenCV runtime needs these shared libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source.
COPY . .

# Persisted at runtime via a volume (see docker-compose.yml).
RUN mkdir -p artifacts/evidence models/weights data/samples

EXPOSE 8501

# Default: launch the analytics dashboard. Override `command` to run the
# pipeline instead, e.g.
#   docker run ... python app.py --video data/samples/test_video.mp4
CMD ["python", "-m", "streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501"]
