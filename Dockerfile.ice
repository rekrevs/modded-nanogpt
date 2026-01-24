FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.12.7
ENV PATH=/usr/local/bin:$PATH

# Install build dependencies
RUN apt update && apt install -y --no-install-recommends build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev \
    && apt clean && rm -rf /var/lib/apt/lists/*

# Build Python from source
RUN curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.tgz

RUN ln -s /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.12 /usr/local/bin/pip

WORKDIR /modded-nanogpt

# Install Python dependencies
COPY requirements.txt /modded-nanogpt/requirements.txt
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Install specific PyTorch version that has Flash Attention 3 kernel support
# This version is explicitly tested in the modded-nanogpt records
RUN pip install torch==2.10.0.dev20251210+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126

# Pre-download Flash Attention 3 kernel to avoid runtime latency
RUN python -c "from kernels import get_kernel; get_kernel('varunneal/flash-attention-3')"

# Copy training code
COPY train_gpt.py triton_kernels.py run.sh /modded-nanogpt/
COPY data/*.py /modded-nanogpt/data/

# Environment for distributed training
ENV NCCL_DEBUG=WARN
ENV NCCL_IB_DISABLE=1

CMD ["bash"]
ENTRYPOINT []
