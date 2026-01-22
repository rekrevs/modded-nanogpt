# Modded-NanoGPT on ICE: The Journey

This document captures the steps taken to set up modded-nanogpt for training on the ICE GPU cluster.

## Goal

Run the modded-nanogpt speedrun on ICE cluster - achieving ≤3.28 validation loss on FineWeb in ~100 seconds using 8xH100 GPUs.

## Phase 1: Initial Exploration

### Understanding the Codebase

Explored `modded-nanogpt` to understand:
- Main training script: `train_gpt.py` with 1600 iterations
- Requires 8x H100 GPUs with `torchrun --nproc_per_node=8`
- Uses Flash Attention 3 via `kernels` package
- Specific PyTorch version: `torch==2.10.0.dev20251210+cu126`
- Data: FineWeb10B tokenized dataset (103 chunks, ~20GB)

### Understanding ICE

Explored `../icemgmt` tools:
- ICE namespace: `aic`
- Available: 32x H100, 87x RTX 2080 Ti, 24x GTX 1080 Ti
- Tools: `ice-train.sh`, `ice-job.sh`, `ice-status.sh`

## Phase 2: Quick Trial Run (Single GPU)

### Created Single-GPU Test Configuration

1. **`run_single_gpu.sh`** - Launcher with `nproc_per_node=1`
2. **`train_gpt_single.py`** - Modified hyperparameters:
   - Reduced batch sizes (1/8th of original)
   - Only 20 iterations (vs 1600)
   - Validation every 10 steps

### Ran Initial Test

```bash
ICE_NAMESPACE=aic ./tools/ice-train.sh create nanogpt-test nvidia-h100 1
```

**Issues encountered:**
1. PyTorch nightly 2.11.x didn't have Flash Attention 3 kernel - had to downgrade to 2.10.x
2. Data path configuration needed adjustment

**Result:** Successfully trained for 20 steps, val_loss dropped from 10.83 → 6.88 (but target is ≤3.28 with full training)

## Phase 3: Building a Custom Docker Image

### Why Custom Image?

- Avoid installing dependencies every run (~2 min)
- Pre-download Flash Attention 3 kernel
- Ensure correct PyTorch version (2.10.0.dev20251210+cu126)

### Initial Approach: Building Python from Source (WRONG)

First Dockerfile built Python 3.12 from source - took forever and was unnecessary.

### Corrected Approach: PyTorch Base Image

Used `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` and upgraded PyTorch:

```dockerfile
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

RUN pip install numpy tqdm huggingface-hub kernels setuptools
RUN pip install torch==2.10.0.dev20251210+cu126 --index-url https://download.pytorch.org/whl/nightly/cu126
RUN python -c "from kernels import get_kernel; get_kernel('varunneal/flash-attention-3')"
```

### Registry Authentication Struggles

**Problem 1:** ICE requires resource requests/limits on all pods
- Solution: Added `cpu: "4"`, `memory: "24Gi"` to Kaniko pod

**Problem 2:** Registry project mismatch
- Tried `aic/modded-nanogpt` - wrong project name
- Correct: `aic-misc/modded-nanogpt` (matching URM's setup)

**Problem 3:** Stale credentials
- `urm-registry-cred` secret existed but was stale
- Solution: Created fresh secret with CLI secret from Harbor UI:
  ```bash
  kubectl create secret docker-registry nanogpt-registry-cred \
    --docker-server=registry.ice.ri.se \
    --docker-username=sverker.janson@ri.se \
    --docker-password=YOUR_CLI_SECRET \
    -n aic
  ```

**Problem 4:** OOMKilled during filesystem snapshot
- 8Gi wasn't enough for snapshot after PyTorch install
- Solution: Increased to 24Gi memory

### Final Build Success

```bash
./ice/build-on-ice.sh
```

Build completed in ~17 minutes. Image pushed to:
```
registry.ice.ri.se/aic-misc/modded-nanogpt:latest
```

## Files Created

| File | Purpose |
|------|---------|
| `ice/Dockerfile` | Image definition |
| `ice/build-image.sh` | Local Docker build script |
| `ice/build-on-ice.sh` | Kaniko build on ICE |
| `ice/train.sh` | Training pod launcher |
| `train_gpt_single.py` | Single-GPU test version |
| `run_single_gpu.sh` | Single-GPU launcher |
| `CLAUDE.md` | Project documentation for Claude |

## Key Learnings

1. **PyTorch version matters** - Flash Attention 3 kernel only built for specific versions
2. **ICE requires strict resource requests** - limits must equal requests
3. **Harbor registry auth is per-project** - can't push to new repos without proper permissions
4. **Kaniko needs lots of memory** - 24Gi needed for PyTorch image snapshots
5. **Use existing base images** - don't build Python from source

## Phase 4: Full 8-GPU Training

### Training Run

```bash
./ice/train.sh speedrun 8 nvidia-h100
kubectl cp . aic/train-speedrun:/workspace/modded-nanogpt
kubectl exec train-speedrun -n aic -- bash -c "cd /workspace/modded-nanogpt/data && python cached_fineweb10B.py"
kubectl exec train-speedrun -n aic -- bash -c "cd /workspace/modded-nanogpt && ./run.sh"
```

### Results

| Metric | Target | Achieved |
|--------|--------|----------|
| **Validation Loss** | ≤3.28 | **3.2761** ✓ |
| **Training Time** | ~100s | 222s (~3.7 min) |
| **Steps** | 1600 | 1600 |

Training time was slower than the ideal ~100 seconds due to NCCL communication overhead on ICE cluster network (vs NVLink in original benchmarks).

## Phase 5: Persistent Data Storage

### Problem
FineWeb data (~20GB, 104 files) had to be re-downloaded for every training run.

### Solution
Created a PersistentVolumeClaim to store data across runs:

```yaml
# ice/nanogpt-data-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nanogpt-data
  namespace: aic
spec:
  accessModes: [ReadWriteOnce]
  resources:
    requests:
      storage: 50Gi
  storageClassName: rook-ceph-rbd
```

Data is now mounted at `/data` in training pods and persists across runs.

### Updated Workflow

```bash
# First time only - download data
./ice/train.sh dataload 1 nvidia-h100
kubectl cp . aic/train-dataload:/workspace/modded-nanogpt
kubectl exec train-dataload -n aic -- bash -c "cd /workspace/modded-nanogpt/data && DATA_PATH=/data python cached_fineweb10B.py"
kubectl delete pod train-dataload -n aic

# Subsequent runs - data is already there
./ice/train.sh speedrun 8 nvidia-h100
kubectl cp . aic/train-speedrun:/workspace/modded-nanogpt
kubectl exec train-speedrun -n aic -- bash -c "cd /workspace/modded-nanogpt && DATA_PATH=/data ./run.sh"
```

## Phase 6: Local Inference on MPS

Created inference scripts to run the trained model locally on Apple Silicon (MPS):

- `inference.py` - Main inference script replacing Flash Attention 3 with PyTorch SDPA
- `extract_weights.py` - Extracts model weights from training checkpoint
- `debug_inference.py` - Debugging utilities

**Note:** Model was trained for perplexity benchmarking, not generation quality. Produces repetitive text.

## Files Created

| File | Purpose |
|------|---------|
| `ice/Dockerfile` | Image definition |
| `ice/build-image.sh` | Local Docker build script |
| `ice/build-on-ice.sh` | Kaniko build on ICE |
| `ice/train.sh` | Training pod launcher |
| `ice/nanogpt-data-pvc.yaml` | Persistent volume for FineWeb data |
| `inference.py` | Local inference on MPS |
| `extract_weights.py` | Checkpoint weight extraction |
| `train_gpt_single.py` | Single-GPU test version |
| `run_single_gpu.sh` | Single-GPU launcher |
| `CLAUDE.md` | Project documentation |

## Key Learnings

1. **PyTorch version matters** - Flash Attention 3 kernel only built for specific versions
2. **ICE requires strict resource requests** - limits must equal requests
3. **Harbor registry auth is per-project** - can't push to new repos without proper permissions
4. **Kaniko needs lots of memory** - 24Gi needed for PyTorch image snapshots
5. **Use existing base images** - don't build Python from source
6. **Persistent storage saves time** - avoid re-downloading 20GB data each run
7. **SDPA can replace FA3** - for inference on non-CUDA hardware
