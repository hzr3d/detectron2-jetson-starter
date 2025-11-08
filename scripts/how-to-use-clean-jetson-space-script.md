If Jetson freezes or crashes when building Detector2, it is mostly like disc space or memory is exhausted. So need to clean up space on the Jetson. Follow the below steps:

Copy it to your Jetson and run a safe dry-run first:

bash clean_jetson_space.sh --dry-run


Execute cleanup (with confirmation prompt):

bash clean_jetson_space.sh


Non-interactive + include Docker prune + create 8 GB swap:

bash clean_jetson_space.sh -y --docker --swap 8G


What it does:

Cleans APT caches (/var/cache/apt/archives, /var/lib/apt/lists)

Removes Python/pip/torch caches

Deletes detectron2/build if present

Vacuums systemd journal to 100 MB

Removes NVIDIA local repo caches (safe):
/var/l4t-cuda-tegra-repo-ubuntu2204-12-6-local,
/var/cudnn-local-tegra-repo-ubuntu2204-9.3.0,
/var/nv-tensorrt-local-tegra-repo-ubuntu2204-10.3.0-cuda-12.5

(Optional) --docker prunes unused Docker images/containers

(Optional) --swap 8G creates/activates /swapfile and adds it to /etc/fstab

Prints “before/after” disk usage and estimated space freed

After running it, verify:

df -h /
free -h





**** Manual steps tp clean up space ****

🧭 Step 1 — Check disk usage first
df -h


Look at the line for / (root).
If Use% is > 90 %, you’re out of space — that’s the main reason for freezing.

🧮 Step 2 — Find the biggest directories
sudo du -h --max-depth=1 / | sort -hr | head -20
sudo du -h --max-depth=1 /usr | sort -hr | head -20
sudo du -h --max-depth=1 /var | sort -hr | head -20
sudo du -h --max-depth=1 /home | sort -hr | head -20


This shows which folders (e.g., /usr, /var/lib/docker, /home) are using the most space.

🧹 Step 3 — Free disk space safely
🔧 Clean APT package cache
sudo apt-get clean
sudo rm -rf /var/cache/apt/archives/*
sudo rm -rf /var/lib/apt/lists/*

🐳 Clean Docker data (this usually frees several GB)
sudo docker system df
sudo docker system prune -a


⚠️ This removes all stopped containers and unused images.
You’ll rebuild your container next time with ./docker/run.sh.

🧩 Remove leftover build artifacts and caches
rm -rf /workspace/detectron2/build
rm -rf ~/.cache/pip
rm -rf ~/.torch
rm -rf ~/.cache/torch

🧱 Remove old log or crash files
sudo journalctl --vacuum-size=100M
sudo rm -rf /var/crash/*

🗑️ Remove large archives
find / -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tgz" \) -size +100M -exec ls -lh {} \;


Delete anything you don’t need.

💾 Step 4 — Verify free space
df -h


Make sure / now has at least 8 – 10 GB free before building Detectron2 again.

🧠 Step 5 — Add swap (to prevent memory freezes)

Jetson devices often freeze from low RAM, not just disk.
You can safely add an 8 GB swapfile:

sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
swapon --show


To make it permanent:

echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab


✅ This doesn’t replace RAM, but it prevents hard freezes by letting the compiler spill memory to disk.

🧰 Step 6 — Re-run the build

Once you have enough disk and swap:

cd /workspace
export PIP_NO_BUILD_ISOLATION=1
D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh
