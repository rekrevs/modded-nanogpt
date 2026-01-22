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

# Download data (103 chunks for full training, or pass smaller number for testing)
cd /workspace/modded-nanogpt/data && python cached_fineweb10B.py [num_chunks]

# Run training
cd /workspace/modded-nanogpt && ./run.sh
```

### Key Files

| File | Purpose |
|------|---------|
| `ice/Dockerfile` | Image with PyTorch 2.10.0+cu126, Flash Attention 3 |
| `ice/build-on-ice.sh` | Build image using Kaniko on ICE |
| `ice/build-image.sh` | Build image locally with Docker |
| `ice/train.sh` | Launch training pods |
| `train_gpt.py` | Main training script (8 GPU) |
| `train_gpt_single.py` | Single-GPU test version |
| `run.sh` | Training launcher (8 GPU) |
| `run_single_gpu.sh` | Single-GPU launcher |

## Dependencies

- **PyTorch**: `torch==2.10.0.dev20251210+cu126` (specific version required for Flash Attention 3 kernel)
- **Flash Attention 3**: Via `kernels` package from `varunneal/flash-attention-3`
- **Data**: FineWeb10B tokenized dataset from `kjj0/fineweb10B-gpt2`

## Training Configuration

Full training (8xH100):
- 1600 iterations
- Batch size schedule: 131K → 262K → 393K tokens
- Target: ≤3.28 validation loss
- Time: ~100 seconds

Single-GPU test (`train_gpt_single.py`):
- 20 iterations
- Reduced batch sizes (1/8th)
- For validation only
