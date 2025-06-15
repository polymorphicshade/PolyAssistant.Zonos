FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt update && \
    apt install -y python3-pip ninja-build build-essential git espeak-ng && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install uv packaging wheel setuptools
RUN uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
#RUN uv pip install --system flash-attn==2.5.0 --no-build-isolation
RUN uv pip install --system flash-attn --no-cache-dir --no-build-isolation --find-links https://download.pytorch.org/whl/cu121/torch_stable.html
RUN uv pip install --system Flask

WORKDIR /app
COPY . ./

RUN uv pip install --system -e . && uv pip install --system -e .[compile]