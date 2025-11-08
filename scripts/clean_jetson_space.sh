#!/usr/bin/env bash
# Clean up disk space on Jetson and (optionally) add swap to avoid build freezes.
# Usage:
#   bash clean_jetson_space.sh -y [--docker] [--swap 8G]
# Options:
#   -y             Run without interactive confirmation (non-interactive).
#   --docker       Also prune Docker images/containers (frees several GB).
#   --swap SIZE    Create/enable a swapfile of SIZE (e.g., 8G). Adds to /etc/fstab if not present.
#   --dry-run      Show what would be removed without deleting.
#   --no-nvidia    Do not remove NVIDIA local repo caches under /var/*-local-tegra-repo-*
#   --help         Show this help.
#
# Safe defaults: APT/pip caches, torch caches, detectron2 build dir, logs. NVIDIA local repo caches
# are also safe to remove; pass --no-nvidia to skip them. Docker prune is opt-in via --docker.

set -euo pipefail

DRY_RUN=0
PRUNE_DOCKER=0
AUTO_YES=0
SWAP_SIZE=""
SKIP_NVIDIA=0

while (( "$#" )); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --docker) PRUNE_DOCKER=1; shift ;;
    --no-nvidia) SKIP_NVIDIA=1; shift ;;
    --swap) SWAP_SIZE="${2:-}"; shift 2 ;;
    -y) AUTO_YES=1; shift ;;
    --help|-h)
      sed -n '1,50p' "$0"; exit 0 ;;
    *)
      echo "Unknown argument: $1"; exit 1;;
  esac
done

confirm() {
  if [ "$AUTO_YES" -eq 1 ]; then return 0; fi
  read -r -p "$1 [y/N]: " ans || true
  case "$ans" in [yY][eE][sS]|[yY]) return 0 ;; *) return 1 ;; esac
}

run() {
  echo "+ $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    eval "$@"
  fi
}

bytes_to_human() {
  awk 'function human(x){ s="BKMGTPEZY"; while (x>=1024 && length(s)>1){x/=1024; s=substr(s,2)} return sprintf("%.1f%s", x, substr(s,1,1)) } {print human($1)}'
}

ROOT_LINE=$(df -B1 / | tail -1)
ROOT_USED=$(echo "$ROOT_LINE" | awk '{print $3}')
ROOT_AVAIL=$(echo "$ROOT_LINE" | awk '{print $4}')
echo "Before cleanup:  Used=$(echo "$ROOT_USED" | bytes_to_human)  Avail=$(echo "$ROOT_AVAIL" | bytes_to_human)"

echo "Scanning large directories (top-level)..."
sudo du -h --max-depth=1 / | sort -hr | head -20 || true

# Targets (present if they exist)
APT_CACHE="/var/cache/apt/archives"
APT_LISTS="/var/lib/apt/lists"
PIP_CACHE="$HOME/.cache/pip"
TORCH_CACHE1="$HOME/.torch"
TORCH_CACHE2="$HOME/.cache/torch"
D2_BUILD="/workspace/detectron2/build"
JOURNALCTL_LIMIT="100M"

NVIDIA_REPOS=(
  "/var/l4t-cuda-tegra-repo-ubuntu2204-12-6-local"
  "/var/cudnn-local-tegra-repo-ubuntu2204-9.3.0"
  "/var/nv-tensorrt-local-tegra-repo-ubuntu2204-10.3.0-cuda-12.5"
)

echo "Planned actions:"
echo " - Clean apt caches: $APT_CACHE and $APT_LISTS"
echo " - Remove pip/torch caches: $PIP_CACHE $TORCH_CACHE1 $TORCH_CACHE2"
echo " - Remove Detectron2 build dir (if exists): $D2_BUILD"
echo " - Vacuum systemd journal to $JOURNALCTL_LIMIT"
if [ "$SKIP_NVIDIA" -eq 0 ]; then
  for d in "${NVIDIA_REPOS[@]}"; do echo " - Remove NVIDIA local repo (if exists): $d"; done
else
  echo " - Skipping NVIDIA local repo cache removal (--no-nvidia)"
fi
if [ "$PRUNE_DOCKER" -eq 1 ]; then
  echo " - Docker prune: remove unused images/containers"
else
  echo " - Docker prune: SKIPPED (use --docker to enable)"
fi
if [ -n "$SWAP_SIZE" ]; then
  echo " - Create/enable swapfile: $SWAP_SIZE at /swapfile"
fi

if ! confirm "Proceed with cleanup?"; then
  echo "Aborted."; exit 1
fi

echo "Cleaning apt cache..."
run "sudo apt-get clean"
run "sudo rm -rf '$APT_CACHE'/*"
run "sudo rm -rf '$APT_LISTS'/*"

echo "Removing Python/pip/torch caches..."
run "rm -rf '$PIP_CACHE' '$TORCH_CACHE1' '$TORCH_CACHE2'"

if [ -d "$D2_BUILD" ]; then
  echo "Removing Detectron2 build artifacts..."
  run "rm -rf '$D2_BUILD'"
fi

if [ "$SKIP_NVIDIA" -eq 0 ]; then
  echo "Removing NVIDIA local repo caches (safe to delete)..."
  for d in "${NVIDIA_REPOS[@]}"; do
    if [ -d "$d" ]; then run "sudo rm -rf '$d'"; fi
  done
fi

echo "Vacuuming systemd journal to ${JOURNALCTL_LIMIT}..."
run "sudo journalctl --vacuum-size=${JOURNALCTL_LIMIT}" || true

if [ "$PRUNE_DOCKER" -eq 1 ]; then
  echo "Docker disk usage before:"
  run "sudo docker system df || true"
  echo "Pruning Docker (images/containers/networks not in use)..."
  run "sudo docker system prune -a -f || true"
fi

if [ -n "$SWAP_SIZE" ]; then
  echo "Configuring swap at /swapfile ($SWAP_SIZE)..."
  if ! sudo swapon --show | grep -q '/swapfile'; then
    run "sudo fallocate -l '$SWAP_SIZE' /swapfile"
    run "sudo chmod 600 /swapfile"
    run "sudo mkswap /swapfile"
    run "sudo swapon /swapfile"
    if ! grep -q '^/swapfile ' /etc/fstab; then
      run "echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null"
    fi
  else
    echo "Swapfile already active."
  fi
  echo "Active swap:"
  run "swapon --show || true"
fi

ROOT_LINE2=$(df -B1 / | tail -1)
ROOT_USED2=$(echo "$ROOT_LINE2" | awk '{print $3}')
FREED=$(( ROOT_USED - ROOT_USED2 ))
echo "After cleanup:   Used=$(echo "$ROOT_USED2" | bytes_to_human)  Avail=$(df -h / | tail -1 | awk '{print $4}')"
echo "Freed approximately: $(echo "$FREED" | bytes_to_human)"
echo "Done."
