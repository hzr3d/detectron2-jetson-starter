#!/usr/bin/env bash
set -e
NAME=detectron2
IMAGE=dustynv/l4t-pytorch:r36.4.0

if sudo docker ps -a --format '{{.Names}}' | grep -q "^${NAME}$"; then
  echo "Reattaching to existing container ${NAME}..."
  sudo docker start -ai ${NAME}
else
  echo "Creating new container ${NAME}..."
  sudo docker run -it --name ${NAME} --runtime nvidia --network host \
    -v $(pwd):/workspace -w /workspace ${IMAGE} /bin/bash
fi

