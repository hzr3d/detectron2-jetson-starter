#!/usr/bin/env bash
set -euo pipefail

IMAGE="dustynv/l4t-pytorch"
if command -v autotag >/dev/null 2>&1; then
  IMAGE_TAG="$(autotag l4t-pytorch)"
else
  IMAGE_TAG="${L4T_TAG:-r36.4.0}"
fi

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
if [ ! -f "$XAUTH" ]; then
  xauth nlist :0 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge - || true
  chmod a+r $XAUTH || true
fi

echo "Launching ${IMAGE}:${IMAGE_TAG} ..."
sudo docker run -it --rm --runtime nvidia --network host   -e DISPLAY=$DISPLAY   -e XAUTHORITY=${XAUTH}   -v ${XSOCK}:${XSOCK}   -v ${XAUTH}:${XAUTH}   -e QT_X11_NO_MITSHM=1   -v /dev:/dev   -v /tmp:/tmp   -v $(pwd):/workspace   --workdir /workspace   ${IMAGE}:${IMAGE_TAG} /bin/bash
