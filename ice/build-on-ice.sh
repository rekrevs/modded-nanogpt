#!/bin/bash
# Build modded-nanogpt image on ICE using Kaniko
# Usage: ./ice/build-on-ice.sh [tag]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ICE_NAMESPACE="aic"
REGISTRY="registry.ice.ri.se"
IMAGE_NAME="aic-misc/modded-nanogpt"
TAG="${1:-latest}"
FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
REGISTRY_SECRET="nanogpt-registry-cred"

POD_NAME="nanogpt-build-image"

echo "=== Building Modded-NanoGPT Image on ICE ==="
echo "Target: ${FULL_IMAGE}"
echo "Namespace: ${ICE_NAMESPACE}"
echo ""

# Check if build pod exists
if kubectl get pod "$POD_NAME" -n "$ICE_NAMESPACE" &>/dev/null; then
    echo "Build pod already exists. Deleting..."
    kubectl delete pod "$POD_NAME" -n "$ICE_NAMESPACE" --wait
fi

# Create build context tarball
echo "Creating build context..."
cd "$SCRIPT_DIR/.."
tar -czf /tmp/nanogpt-build-context.tar.gz \
    --exclude='.git' \
    --exclude='data/fineweb*' \
    --exclude='logs' \
    --exclude='records' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pt' \
    --exclude='*.bin' \
    ice/Dockerfile \
    requirements.txt \
    train_gpt.py \
    triton_kernels.py \
    run.sh \
    data/*.py

echo "Context size: $(du -h /tmp/nanogpt-build-context.tar.gz | cut -f1)"

echo ""
echo "Starting Kaniko build pod..."
cat <<EOF | kubectl apply -n "$ICE_NAMESPACE" -f -
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
spec:
  containers:
  - name: kaniko
    image: gcr.io/kaniko-project/executor:debug
    resources:
      requests:
        cpu: "4"
        memory: "24Gi"
      limits:
        memory: "24Gi"
    command: ["/busybox/sh", "-c"]
    args:
    - |
      echo "Waiting for build context..."
      while [ ! -f /workspace/context.tar.gz ]; do sleep 1; done
      echo "Build context found, starting build..."
      /kaniko/executor \
        --dockerfile=ice/Dockerfile \
        --context=tar:///workspace/context.tar.gz \
        --destination=${FULL_IMAGE} \
        --cache=true
    volumeMounts:
    - name: build-context
      mountPath: /workspace
    - name: docker-config
      mountPath: /kaniko/.docker
  restartPolicy: Never
  volumes:
  - name: build-context
    emptyDir: {}
  - name: docker-config
    secret:
      secretName: ${REGISTRY_SECRET}
      items:
      - key: .dockerconfigjson
        path: config.json
EOF

echo ""
echo "Waiting for pod to initialize..."
kubectl wait --for=condition=Ready pod/$POD_NAME -n "$ICE_NAMESPACE" --timeout=120s

echo "Copying build context to pod..."
kubectl cp /tmp/nanogpt-build-context.tar.gz "$ICE_NAMESPACE/$POD_NAME:/workspace/context.tar.gz"

echo ""
echo "Build started! Monitor with:"
echo "  kubectl logs -f $POD_NAME -n $ICE_NAMESPACE"
echo ""
echo "When complete:"
echo "  kubectl delete pod $POD_NAME -n $ICE_NAMESPACE"
echo ""
echo "Then use the image:"
echo "  ./ice/train.sh speedrun 8 nvidia-h100"
