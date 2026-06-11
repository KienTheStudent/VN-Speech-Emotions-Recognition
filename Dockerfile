FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.docker.txt /tmp/requirements.docker.txt
RUN python -m pip install --no-cache-dir "setuptools>=68" wheel && \
    python -m pip install -r /tmp/requirements.docker.txt

COPY . /workspace

CMD ["bash"]
