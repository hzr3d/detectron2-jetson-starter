For maintenance and persistence tips, see [JETSON_MAINTENANCE.md](JETSON_MAINTENANCE.md).

# Detectron2 on Jetson Orin Nano (JetPack R36.4.x, cuDNN 9.3)

This project provides a **ready-to-run container setup** for Facebook AI Research’s **Detectron2** on **Jetson Orin Nano**, running **JetPack 6.1 (Jetson Linux 36.4.x)** with **CUDA 12.6** and **cuDNN 9.3 (9.3.0.75-1)**.

It supports:
- Running pre-trained COCO instance segmentation and object detection models.
- Viewing bounding boxes, masks, and labels.
- Fine-tuning or retraining Detectron2 on your own dataset (for example, flowers).

---

## 🧰 System Requirements
- Jetson Orin Nano (or any Orin/Orin NX)
- JetPack **R36.4.x**
- Docker installed (`sudo apt-get install -y docker.io`)
- Internet access
- ~6 GB free disk space

---

## 🚀 Quick Setup

### 1️⃣ Unzip & enter project
```bash
unzip detectron2-jetson-starter-patched.zip
cd detectron2-jetson-starter
```

### 2️⃣ Launch the Jetson PyTorch container
```bash
./docker/run.sh
```
This opens an interactive shell with GPU and camera access.

---

## 🧹 Fix Python mirrors & dependencies
Inside the container:
```bash
bash /workspace/scripts/fix_pip.sh
```
This removes Jetson pip mirrors, sets `pypi.org`, installs `python3-opencv`, and prepares dependencies.

---

## 🏗️ Build & install Detectron2
```bash
D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh
```
✅ Builds Detectron2 for SM 8.7 GPU arch  
✅ Uses CUDA 12–compatible fork  
✅ Verifies import

🕑 Takes ~20–25 minutes the first time.

---

## 🧪 Run the demo
### Image mode
```bash
python3 demo/infer.py --input demo/sample.jpg --output demo/out.jpg
```
or your own:
```bash
python3 demo/infer.py --input path/to/image.jpg --output demo/out.jpg
```
### Webcam
```bash
python3 demo/infer.py --webcam 0
```

`demo/out.jpg` will show boxes, labels, and masks.

---

## 🧩 Where labels come from
Detectron2 uses **COCO’s 80 classes**, defined in:
```
/workspace/detectron2/detectron2/data/datasets/builtin_meta.py
```
Inspect them:
```python
from detectron2.data import MetadataCatalog
print(MetadataCatalog.get("coco_2017_train").thing_classes)
```
Common ones: `person`, `car`, `dog`, `cat`, `chair`, `bottle`, `tv`, `potted plant`, etc.  
❌ *Flowers are not included*.

---

## 🌸 Fine-tuning / Retraining on your dataset
You can teach Detectron2 to detect your own objects (e.g., flowers).

### 1️⃣ Prepare dataset (COCO format)
```
datasets/flowers/
 ├── train/
 │   ├── rose_001.jpg
 │   └── tulip_002.jpg
 ├── val/
 │   ├── rose_010.jpg
 │   └── tulip_011.jpg
 ├── annotations/
 │   ├── train.json
 │   └── val.json
```

Create annotations with [LabelMe](https://github.com/wkentaro/labelme), [CVAT](https://cvat.org), or [Roboflow](https://roboflow.com).

### 2️⃣ Register dataset
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("flowers_train", {}, "datasets/flowers/annotations/train.json", "datasets/flowers/train")
register_coco_instances("flowers_val", {}, "datasets/flowers/annotations/val.json", "datasets/flowers/val")
```

### 3️⃣ Configure for fine-tuning
```python
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("flowers_train",)
cfg.DATASETS.TEST = ("flowers_val",)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # adjust to your dataset
```

### 4️⃣ Train
```bash
python3 train_net.py --config-file your_config.yaml --num-gpus 1
```
or in Python:
```python
from detectron2.engine import DefaultTrainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### 5️⃣ Use your trained weights
```python
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
```

Then rerun the demo.

---

## ✅ Quick Commands Summary

| Task | Command |
|------|----------|
| Launch container | `./docker/run.sh` |
| Fix pip mirrors | `bash /workspace/scripts/fix_pip.sh` |
| Build Detectron2 | `D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh` |
| Run demo | `python3 demo/infer.py --input demo/sample.jpg --output demo/out.jpg` |
| Check COCO labels | `python3 -c "from detectron2.data import MetadataCatalog; print(MetadataCatalog.get('coco_2017_train').thing_classes)"` |
| Fine-tune custom dataset | See “Fine-tuning” section |

---

## 📚 References
- [Detectron2 GitHub](https://github.com/facebookresearch/detectron2)
- [COCO Dataset](https://cocodataset.org)
- [Detectron2 Custom Datasets Guide](https://detectron2.readthedocs.io/tutorials/datasets.html)
- [NVIDIA Jetson Containers](https://github.com/dusty-nv/jetson-containers)
- [Mask R-CNN Paper (He et al., 2017)](https://arxiv.org/abs/1703.06870)

---

Happy experimenting with Detectron2 on Jetson 🚀
