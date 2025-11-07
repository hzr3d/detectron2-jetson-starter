#!/usr/bin/env bash
set -euo pipefail

echo "===> Forcing pip to use pypi.org and disabling extra mirrors"
rm -f /etc/pip.conf /etc/pip/pip.conf 2>/dev/null || true
rm -f "${HOME}/.config/pip/pip.conf" "${HOME}/.pip/pip.conf" 2>/dev/null || true

export PIP_INDEX_URL="https://pypi.org/simple"
unset PIP_EXTRA_INDEX_URL
export PIP_DEFAULT_TIMEOUT=35

echo "===> Installing certificates and OpenCV via apt (ARM64-safe)"
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates python3-opencv
update-ca-certificates || true

echo "===> Verifying pip config"
python3 -m pip config list || true

echo "===> Preinstall build deps via pip (hitting pypi.org)"
python3 -m pip install --no-cache-dir --index-url https://pypi.org/simple --timeout 40 --retries 1 \
  wheel ninja yacs tabulate termcolor cloudpickle pyyaml iopath>=0.1.9 fvcore>=0.1.5.post20221221 Pillow matplotlib tqdm

echo "===> Done. Now run: D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh"
