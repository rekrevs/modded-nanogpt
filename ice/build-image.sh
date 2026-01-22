#!/bin/bash
# Build and push modded-nanogpt training image to ICE registry
# Usage: ./ice/build-image.sh [tag]

set -e

REGISTRY="registry.ice.ri.se"
IMAGE_NAME="aic-misc/modded-nanogpt"
TAG="${1:-latest}"

FULL_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "=== Building Modded-NanoGPT Training Image ==="
echo "Image: ${FULL_IMAGE}"
echo ""

cd "$(dirname "$0")"

# Build the image
echo "Building image (this may take 10-15 minutes)..."
docker build -t "${FULL_IMAGE}" -f Dockerfile ..

echo ""
echo "Build complete!"
echo ""

# Push to registry
echo "Pushing to ${REGISTRY}..."
echo "(You may need to run: docker login ${REGISTRY})"
docker push "${FULL_IMAGE}"

echo ""
echo "=== Done ==="
echo "Image available at: ${FULL_IMAGE}"
echo ""
echo "To use this image on ICE:"
echo "  kubectl run nanogpt --image=${FULL_IMAGE} --restart=Never ..."
