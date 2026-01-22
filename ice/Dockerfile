# Modded-NanoGPT Training Image for ICE
# Optimized for 8xH100 speedrun training
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    numpy \
    tqdm \
    huggingface-hub \
    kernels \
    setuptools

# Upgrade to specific PyTorch version with Flash Attention 3 kernel support
RUN pip install --no-cache-dir \
    torch==2.10.0.dev20251210+cu126 \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Pre-download Flash Attention 3 kernel to avoid runtime latency
RUN python -c "from kernels import get_kernel; get_kernel('varunneal/flash-attention-3')"

# Environment for distributed training
ENV NCCL_DEBUG=WARN
ENV PYTORCH_ALLOC_CONF=expandable_segments:True

# Verify installations
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
RUN python -c "from kernels import get_kernel; fa3 = get_kernel('varunneal/flash-attention-3'); print('Flash Attention 3: OK')"

CMD ["/bin/bash"]
