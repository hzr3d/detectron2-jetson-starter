# Jetson Maintenance and Persistence Guide

This guide explains how to:
- Prevent rebuilding Detectron2 after every reboot.
- Free up disk space and prevent freezes during builds.

---

## 🧱 Avoid Rebuilding Detectron2 After Reboot

When you reboot Jetson, Docker containers stop but remain saved. You can reuse the same container so you don’t have to rebuild Detectron2 every time.

### 1️⃣ Create a named container (only once)
```bash
cd detectron2-jetson-starter
sudo docker run -it --name detectron2 --runtime nvidia --network host \
  -v $(pwd):/workspace -w /workspace dustynv/l4t-pytorch:r36.4.0 /bin/bash
Then inside the container:

bash
Copy code
bash /workspace/scripts/fix_pip.sh
export PIP_NO_BUILD_ISOLATION=1
D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh
After it finishes, exit the container (exit or Ctrl+D).

2️⃣ After reboot
Reattach to the same container:

bash
Copy code
sudo docker start -ai detectron2
You’re back with Detectron2 already built — no rebuild needed.

3️⃣ Optional: Auto-reuse container in docker/run.sh
Replace your docker/run.sh with:

bash
Copy code
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
Now ./docker/run.sh will automatically reuse the same container.

💾 Freeing Up Disk Space & Preventing Freezes
Jetson devices have limited disk space and RAM. A full disk or low memory causes freezes during Detectron2 builds.

🧮 Check your usage
bash
Copy code
df -h /
free -h
If / shows >90% used, or memory is <1 GB free, clean up before rebuilding.

🧹 Automated cleanup script
Use the included script:

bash
Copy code
bash clean_jetson_space.sh -y --docker --swap 8G
What it does:

Cleans APT caches and package lists.

Removes pip and torch caches.

Deletes old Detectron2 build artifacts.

Removes NVIDIA local repo installers.

Prunes unused Docker images (optional with --docker).

Adds swap (e.g., 8G) to prevent memory exhaustion.

Prints before/after disk usage.

Preview without deleting:

bash
Copy code
bash clean_jetson_space.sh --dry-run
🧠 When Rebuild Is Actually Needed
Situation	Rebuild Needed?
Jetson reboot	❌ No
Restart container	❌ No
Container deleted (docker system prune -a)	✅ Yes
JetPack or PyTorch updated	✅ Yes

✅ Quick Reference Commands
Task	Command
Start container	sudo docker start -ai detectron2
Rebuild only if needed	D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh
Check disk	df -h /
Run cleanup + swap	bash clean_jetson_space.sh -y --docker --swap 8G

🧩 Notes
Always ensure 8–10 GB free before building Detectron2.

If you see Building editable for detectron2... and Jetson freezes, run cleanup + enable swap first.

If you remove the container/image with docker system prune -a, you’ll need to rebuild once.

Enjoy faster, stable Detectron2 development on Jetson 🚀

