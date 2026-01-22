#!/bin/bash
# Launch modded-nanogpt training on ICE
# Usage: ./ice/train.sh <name> [gpu-count] [gpu-type]
#
# Examples:
#   ./ice/train.sh speedrun 8 nvidia-h100    # Full 8xH100 speedrun
#   ./ice/train.sh test 1 nvidia-h100        # Single GPU test

set -e

NAME="${1:-nanogpt-train}"
GPU_COUNT="${2:-8}"
GPU_TYPE="${3:-nvidia-h100}"

NAMESPACE="aic"
IMAGE="registry.ice.ri.se/aic-misc/modded-nanogpt:latest"
REGISTRY_SECRET="nanogpt-registry-cred"

# Calculate resources (4 CPU cores + 16GB per GPU)
CPU_REQUEST=$((GPU_COUNT * 4))
MEM_REQUEST="$((GPU_COUNT * 16))Gi"

POD_NAME="train-${NAME}"

echo "=== Modded-NanoGPT Training ==="
echo "Name: ${POD_NAME}"
echo "GPUs: ${GPU_COUNT}x ${GPU_TYPE}"
echo "Image: ${IMAGE}"
echo "Namespace: ${NAMESPACE}"
echo ""

# Check if pod already exists
if kubectl get pod "$POD_NAME" -n "$NAMESPACE" &>/dev/null; then
    echo "Pod ${POD_NAME} already exists!"
    echo "To delete: kubectl delete pod ${POD_NAME} -n ${NAMESPACE}"
    exit 1
fi

# Create data PVC if it doesn't exist
if ! kubectl get pvc nanogpt-data -n "$NAMESPACE" &>/dev/null; then
    echo "Creating persistent volume for FineWeb data..."
    kubectl apply -f "$(dirname "$0")/nanogpt-data-pvc.yaml"
fi

# Create the training pod
cat <<EOF | kubectl apply -n "$NAMESPACE" -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  labels:
    app: modded-nanogpt
    experiment: ${NAME}
spec:
  restartPolicy: Never
  imagePullSecrets:
  - name: ${REGISTRY_SECRET}
  containers:
  - name: training
    image: ${IMAGE}
    command: ["sleep", "infinity"]
    resources:
      requests:
        cpu: "${CPU_REQUEST}"
        memory: "${MEM_REQUEST}"
        nvidia.com/gpu: "${GPU_COUNT}"
      limits:
        memory: "${MEM_REQUEST}"
        nvidia.com/gpu: "${GPU_COUNT}"
    env:
    - name: NCCL_DEBUG
      value: "WARN"
    - name: NCCL_IB_DISABLE
      value: "1"
    - name: MASTER_ADDR
      value: "localhost"
    - name: MASTER_PORT
      value: "29500"
    volumeMounts:
    - name: dshm
      mountPath: /dev/shm
    - name: workspace
      mountPath: /workspace
    - name: data
      mountPath: /data
  nodeSelector:
    accelerator: "${GPU_TYPE}"
  volumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 64Gi
  - name: workspace
    emptyDir: {}
  - name: data
    persistentVolumeClaim:
      claimName: nanogpt-data
EOF

echo "Waiting for pod to start..."
kubectl wait --for=condition=Ready pod/${POD_NAME} -n ${NAMESPACE} --timeout=300s

echo ""
echo "=== Pod Ready ==="
echo ""
echo "Data volume mounted at /data (persistent across runs)"
echo ""
echo "Next steps:"
echo ""
echo "1. Sync code to pod:"
echo "   kubectl cp . ${NAMESPACE}/${POD_NAME}:/workspace/modded-nanogpt"
echo ""
echo "2. Download data (only needed once, stored in /data):"
echo "   kubectl exec -it ${POD_NAME} -n ${NAMESPACE} -- bash"
echo "   cd /workspace/modded-nanogpt/data && DATA_PATH=/data python cached_fineweb10B.py"
echo ""
echo "3. Run training (with persistent data):"
echo "   cd /workspace/modded-nanogpt && DATA_PATH=/data ./run.sh"
echo ""
echo "4. Stop pod when done:"
echo "   kubectl delete pod ${POD_NAME} -n ${NAMESPACE}"
