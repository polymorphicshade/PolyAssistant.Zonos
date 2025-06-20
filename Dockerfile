FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

RUN apt update && \
    apt install -y python3-pip ninja-build build-essential git espeak-ng && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install uv packaging wheel setuptools
RUN uv pip install --system torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
RUN uv pip install --system flash-attn --no-cache-dir --no-build-isolation --find-links https://download.pytorch.org/whl/cu121/torch_stable.html
RUN uv pip install --system Flask

WORKDIR /app
COPY . ./

RUN uv pip install --system -e . && uv pip install --system -e .[compile]