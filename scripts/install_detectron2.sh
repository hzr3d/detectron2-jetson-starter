#!/usr/bin/env bash
set -euo pipefail

# Build Detectron2 on Jetson Orin (SM 8.7) inside the l4t-pytorch container.

D2_DIR="/workspace/detectron2"
D2_REF="${D2_REF:-}"         # leave empty by default; use main or fork
D2_USE_COMMUNITY_FORK="${D2_USE_COMMUNITY_FORK:-1}"  # default ON
CLEAN_PIP="${CLEAN_PIP:-1}"

if [ "${CLEAN_PIP}" = "1" ]; then
  echo "===> Forcing pip to use pypi.org (bypassing mirrors)"
  rm -f /etc/pip.conf /etc/pip/pip.conf 2>/dev/null || true
  rm -f "${HOME}/.config/pip/pip.conf" "${HOME}/.pip/pip.conf" 2>/dev/null || true
  export PIP_INDEX_URL="https://pypi.org/simple"
  unset PIP_EXTRA_INDEX_URL
  export PIP_DEFAULT_TIMEOUT=35
fi

echo "===> Python/PyTorch info"
python3 - <<'PY'
import torch, os
print("Python:", os.sys.version)
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a")
PY

echo "===> Install build deps"
apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential cmake git nano unzip wget \
    python3-dev python3-setuptools \
    libglib2.0-0 libx11-6 libgl1 \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

pip3 install --upgrade pip --no-cache-dir --index-url https://pypi.org/simple --timeout 40 --retries 1
pip3 install --no-cache-dir --index-url https://pypi.org/simple --timeout 40 --retries 1 \
    wheel ninja Pillow matplotlib yacs tqdm termcolor cloudpickle pyyaml tabulate "iopath>=0.1.9 fvcore>=0.1.5.post20221221"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.7"

if [ "${D2_USE_COMMUNITY_FORK}" = "1" ]; then
  echo "===> Using community fork for CUDA12 friendliness"
  rm -rf "${D2_DIR}"
  git clone --depth=1 https://github.com/ilovejs/detectron2-better.git "${D2_DIR}"
  cd "${D2_DIR}"
else
  echo "===> Cloning facebookresearch/detectron2"
  rm -rf "${D2_DIR}"
  git clone https://github.com/facebookresearch/detectron2.git "${D2_DIR}"
  cd "${D2_DIR}"
  if [ -n "${D2_REF}" ]; then
    git checkout "${D2_REF}" || echo "Warning: could not checkout D2_REF=${D2_REF}, continuing on main"
  fi
fi

echo "===> Building Detectron2"
PIP_NO_BUILD_ISOLATION=1 python3 -m pip install --no-cache-dir --no-build-isolation -e . --index-url https://pypi.org/simple --timeout 120 --retries 1

echo "===> Verifying import"
python3 - <<'PY'
import detectron2, torch
from detectron2.utils.logger import setup_logger
setup_logger()
print("Detectron2 version:", getattr(detectron2,'__version__','unknown'))
print("Torch:", torch.__version__)
PY

echo "===> Done. Try: python3 demo/infer.py --input demo/sample.jpg --output demo/out.jpg"
