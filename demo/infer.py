import argparse, cv2, time, os
import numpy as np

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

def build_predictor(score_thresh=0.5):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_thresh
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES","") != "-1" else "cpu"
    return DefaultPredictor(cfg)

def run_image(predictor, inp, out):
    img = cv2.imread(inp)
    assert img is not None, f"Could not read image: {inp}"
    outputs = predictor(img)
    v = Visualizer(img[:, :, ::-1], instance_mode=ColorMode.IMAGE)
    out_vis = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cv2.imwrite(out, out_vis)
    print(f"Saved: {out}")

def run_webcam(predictor, dev=0):
    cap = cv2.VideoCapture(dev)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {dev}")
    dt_hist = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time.time()
            outputs = predictor(frame)
            dt_hist.append(time.time()-t0)
            v = Visualizer(frame[:, :, ::-1], instance_mode=ColorMode.IMAGE)
            out_vis = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()[:, :, ::-1]
            cv2.imshow("Detectron2 Webcam", out_vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if dt_hist:
            print(f"Avg inference time: {np.mean(dt_hist)*1000:.1f} ms")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, help="Path to input image")
    ap.add_argument("--output", type=str, default="demo/out.jpg")
    ap.add_argument("--webcam", type=int, help="Camera index to use")
    ap.add_argument("--score", type=float, default=0.5)
    args = ap.parse_args()

    predictor = build_predictor(score_thresh=args.score)

    if args.webcam is not None:
        run_webcam(predictor, args.webcam)
    else:
        assert args.input, "--input is required for image mode"
        run_image(predictor, args.input, args.output)
