For maintenance and persistence tips, see [JETSON_MAINTENANCE.md](JETSON_MAINTENANCE.md).

# Detectron2 on Jetson Orin Nano (JetPack R36.4.x, cuDNN 9.3)

Ready-to-run container setup for **Detectron2** on **Jetson Orin Nano** (JetPack **6.1 / R36.4.x**, CUDA 12.6, cuDNN 9.3).  
Includes: quickstart, where labels come from, fine-tuning, **other pre-trained models**, and **ONNX → TensorRT** export.

---

## 🚀 Quick Setup

**Outside the container** (on Jetson):
```bash
unzip detectron2-jetson-starter-patched.zip
cd detectron2-jetson-starter
./docker/run.sh
```
You should now be **inside** the container (`root@...:/workspace#`).

**Inside the container:**
```bash
# 1) Fix pip mirrors & prep deps
bash /workspace/scripts/fix_pip.sh

# 2) Build Detectron2 (uses CUDA12-friendly fork by default)
D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh
```

### Run a demo
```bash
# Image
python3 demo/infer.py --input demo/sample.jpg --output demo/out.jpg
# Webcam
python3 demo/infer.py --webcam 0
```

`out.jpg` overlays **boxes + labels + masks + scores**. Replace `sample.jpg` with your own image to see actual detections.

---

## 🧩 Where labels come from (COCO)
The default model is COCO **Mask R-CNN R50-FPN 3×**. COCO’s 80 labels live in:
```
/workspace/detectron2/detectron2/data/datasets/builtin_meta.py
```
Print them:
```python
from detectron2.data import MetadataCatalog
print(MetadataCatalog.get("coco_2017_train").thing_classes)
```
> COCO does **not** include “flower” — fine-tune to add it (see below).

---

## 🌸 Fine-tuning / Retraining (custom dataset)

**Dataset layout (COCO JSON):**
```
datasets/flowers/
 ├─ train/              # images
 ├─ val/                # images
 └─ annotations/
     ├─ train.json     # COCO annotations
     └─ val.json
```

**Register & configure:**
```python
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer

register_coco_instances("flowers_train", {}, "datasets/flowers/annotations/train.json", "datasets/flowers/train")
register_coco_instances("flowers_val",   {}, "datasets/flowers/annotations/val.json",   "datasets/flowers/val")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("flowers_train",)
cfg.DATASETS.TEST  = ("flowers_val",)
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 2.5e-4
cfg.SOLVER.MAX_ITER = 1000   # adjust for your dataset size
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3   # e.g., rose, tulip, sunflower

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```
Use your weights:
```python
cfg.MODEL.WEIGHTS = "output/model_final.pth"
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
```

---

## 🧠 Other pre-trained models (beyond COCO Mask R-CNN)

Swap the config path in your script to use any of these:

| Family | Task | Dataset | Config path (pass to `model_zoo.get_config_file`) |
|---|---|---|---|
| **Faster R-CNN** | Object detection | COCO | `COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml` |
| **RetinaNet** | One‑stage detection | COCO | `COCO-Detection/retinanet_R_50_FPN_3x.yaml` |
| **Mask R-CNN** | Instance segm. | COCO | `COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml` |
| **Keypoint R-CNN** | Human pose | COCO Keypoints | `COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml` |
| **Panoptic FPN** | Panoptic segm. | COCO Panoptic | `COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml` |
| **DensePose R-CNN** | Human surface UV | DensePose | `DensePose/densepose_rcnn_R_50_FPN_s1x.yaml` |
| **LVIS Mask R-CNN** | Large‑vocab inst. segm. | LVIS | `LVIS-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml` |
| **Cityscapes** | Urban scene | Cityscapes | `Cityscapes/mask_rcnn_R_50_FPN.yaml` |
| **Pascal VOC** | Detection | VOC | `PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml` |

Usage pattern:
```python
from detectron2.config import get_cfg
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
```

Programmatically list all configs available in your install:
```python
import detectron2, os
path = os.path.join(os.path.dirname(detectron2.__file__), "model_zoo", "configs")
for r,_,fs in os.walk(path):
    for f in fs:
        if f.endswith(".yaml"):
            print(os.path.relpath(os.path.join(r,f), path))
```

> **Jetson tip:** prefer R50 backbones for performance. R101/X101 variants are heavier.

---

## ⚡ Export to ONNX → TensorRT (Jetson)

### 1) Export Detectron2 model to ONNX
The cleanest path is Detectron2’s **TracingAdapter** to make models traceable.

Create `export_onnx.py`:
```python
import torch
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.export import TracingAdapter
from detectron2.engine import DefaultPredictor

# 1) Load model
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.DEVICE = "cuda"
predictor = DefaultPredictor(cfg)
model = predictor.model.eval()

# 2) Sample input (NCHW, float32, 0–255 expected by Detectron2's preprocessing)
H, W = 480, 640
sample = [{"image": torch.randn(3, H, W).cuda() * 255.0, "height": H, "width": W}]

# 3) Wrap with TracingAdapter
adapter = TracingAdapter(model, sample, inference=True)
inputs = adapter.flattened_inputs  # tuple of tensors expected by the adapter

# 4) Export (dynamic shape ready)
torch.onnx.export(
    adapter,                       # model
    inputs,                        # example inputs (tuple)
    "model.onnx",
    opset_version=13,
    input_names=[f"input_{i}" for i in range(len(inputs))],
    output_names=["instances"],
    dynamic_axes={
        f"input_{i}": {0: "batch", -2: "height", -1: "width"} for i in range(len(inputs))
    },
)
print("Saved model.onnx")
```

Run it **inside the container**:
```bash
python3 export_onnx.py
```

> Notes:
> - Some advanced configs may need small tweaks at export time. Start with R50-FPN models.
> - Post‑processing (NMS/formatting) is often not fused into ONNX; handle post‑proc on CPU or port to TensorRT plugins if desired.

### 2) Build a TensorRT engine with `trtexec`
On Jetson (has TensorRT preinstalled):
```bash
# FP16 engine (recommended on Orin)
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=model_fp16.plan \
  --fp16 \
  --workspace=2048 \
  --buildOnly
```

**Dynamic shapes** (if you exported with dynamic axes), set min/opt/max:
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=model_fp16.plan \
  --fp16 --workspace=2048 \
  --minShapes=input_0:1x3x360x640 \
  --optShapes=input_0:1x3x480x800 \
  --maxShapes=input_0:1x3x720x1280
```

**INT8** (needs calibration or QAT weights):
```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=model.onnx \
  --saveEngine=model_int8.plan \
  --int8 --workspace=2048 \
  --calib=calib.cache
```

### 3) Running the TensorRT engine
You can deploy the `.plan` with a simple TensorRT runtime or use **ONNX Runtime-TensorRT**. For quick tests:
```bash
/usr/src/tensorrt/bin/trtexec --loadEngine=model_fp16.plan --shapes=input_0:1x3x480x640
```

**Important**: TensorRT expects **NCHW float32/16** inputs. Keep pre/post‑processing (normalization, resizing, NMS, COCO label mapping) in Python/C++ alongside the engine.

---

## ✅ Quick Commands Summary

| Task | Command |
|---|---|
| Launch container | `./docker/run.sh` |
| Fix pip mirrors | `bash /workspace/scripts/fix_pip.sh` |
| Build Detectron2 | `D2_USE_COMMUNITY_FORK=1 /workspace/scripts/install_detectron2.sh` |
| Run demo | `python3 demo/infer.py --input demo/sample.jpg --output demo/out.jpg` |
| List model configs | *see Python snippet above* |
| Export ONNX | `python3 export_onnx.py` |
| Build TensorRT (FP16) | `trtexec --onnx=model.onnx --saveEngine=model_fp16.plan --fp16` |

---

Happy experimenting on Jetson! 🚀
