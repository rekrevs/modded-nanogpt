# Modded-NanoGPT

GPT-2 speedrun training - target ≤3.28 validation loss on FineWeb in ~100 seconds using 8xH100 GPUs.

## ICE Cluster Setup

### Namespace and Registry

- **Kubernetes namespace**: `aic`
- **Registry**: `registry.ice.ri.se`
- **Registry project**: `aic-misc`
- **Image**: `registry.ice.ri.se/aic-misc/modded-nanogpt:latest`

### Registry Credentials

Registry credentials are stored in Kubernetes secrets:

- `nanogpt-registry-cred` - Project credentials (username: `sverker.janson@ri.se`)

To recreate credentials if needed:
```bash
# Get CLI secret from https://registry.ice.ri.se (User Profile → CLI secret)
kubectl create secret docker-registry nanogpt-registry-cred \
  --docker-server=registry.ice.ri.se \
  --docker-username=sverker.janson@ri.se \
  --docker-password=YOUR_CLI_SECRET \
  -n aic
```

### Building the Image

**Option 1: Build on ICE with Kaniko (no local Docker needed)**
```bash
./ice/build-on-ice.sh [tag]
```

**Option 2: Build locally with Docker**
```bash
colima start  # Start Docker daemon on macOS
docker login registry.ice.ri.se
./ice/build-image.sh [tag]
```

### Persistent Data Storage

FineWeb data (~20GB) and checkpoints are stored in a PersistentVolumeClaim:

- **PVC**: `nanogpt-data` (50Gi, rook-ceph-rbd)
- **Mount point**: `/data` in training pods
- **Data location**: `/data/data/fineweb10B/`
- **Checkpoints**: `/data/checkpoints/{run_id}/`

The PVC is automatically created by `train.sh` if it doesn't exist.

**Current checkpoints on PVC:**
- `94384ee4.../state_step002000.pt` through `state_step016000.pt` (10x training run)

### Running Training

```bash
# Launch 8xH100 for full speedrun
./ice/train.sh speedrun 8 nvidia-h100

# Single GPU test
./ice/train.sh test 1 nvidia-h100
```

Inside the pod:
```bash
# Sync code
kubectl cp . aic/train-speedrun:/workspace/modded-nanogpt

# Download data (only needed once - persists in PVC)
cd /workspace/modded-nanogpt/data && DATA_PATH=/data python cached_fineweb10B.py

# Run training with persistent data
cd /workspace/modded-nanogpt && DATA_PATH=/data ./run.sh
```

### Key Files

| File | Purpose |
|------|---------|
| `ice/Dockerfile` | Image with PyTorch 2.10.0+cu126, Flash Attention 3 |
| `ice/build-on-ice.sh` | Build image using Kaniko on ICE |
| `ice/build-image.sh` | Build image locally with Docker |
| `ice/train.sh` | Launch training pods |
| `ice/nanogpt-data-pvc.yaml` | PVC for persistent FineWeb data |
| `train_gpt.py` | Main training script (8 GPU, 1600 steps) |
| `train_gpt_long.py` | Extended training (16000 steps, checkpoints every 2000) |
| `train_gpt_single.py` | Single-GPU test version |
| `run.sh` | Training launcher (8 GPU) |
| `run_long.sh` | Long training launcher (10x steps) |
| `run_single_gpu.sh` | Single-GPU launcher |

## Local Inference (MPS/CPU)

Run the trained model locally on Apple Silicon or CPU:

```bash
# Extract weights from checkpoint (if needed)
python extract_weights.py checkpoints/state_step001600.pt

# Generate text
python inference.py --prompt "Your prompt here" --max_tokens 100 --device mps
```

**Note:** Model was trained for perplexity benchmarking, not generation quality.

| File | Purpose |
|------|---------|
| `inference.py` | Inference with SDPA (replaces Flash Attention 3) |
| `extract_weights.py` | Extract model weights from training checkpoint |
| `debug_inference.py` | Debugging utilities |

## Dependencies

- **PyTorch**: `torch==2.10.0.dev20251210+cu126` (specific version required for Flash Attention 3 kernel)
- **Flash Attention 3**: Via `kernels` package from `varunneal/flash-attention-3`
- **Data**: FineWeb10B tokenized dataset from `kjj0/fineweb10B-gpt2`

## Training Configuration

### Speedrun (1600 steps)
- Batch size schedule: 131K → 262K → 393K tokens
- Target: ≤3.28 validation loss
- Time: ~100 seconds (ideal), ~220 seconds on ICE

**Results:** val_loss **3.2761** ✓ in 222 seconds

### Extended Training (16000 steps)
- 10x longer for better perplexity
- Checkpoints saved every 2000 steps to `/data/checkpoints/`
- Time: ~37 minutes on 8xH100

**Results:** val_loss **3.2328** in 37 minutes

### Generation Quality
Despite lower perplexity, models trained on FineWeb produce incoherent text:
- 84% probability assigned to `<|endoftext|>` (document boundaries)
- Strong copy mechanisms cause repetition
- **Solution:** Fine-tune on focused corpus (Shakespeare, instructions)

### Single-GPU test
- 20 iterations, reduced batch sizes (1/8th)
- For validation only
