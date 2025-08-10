#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run YOLO inference on images")
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights .pt or pretrained model name")
    p.add_argument("--source", type=str, default="/workspace/data/gpr_yolo/images/val", help="File/dir/glob")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--save_dir", type=str, default="/workspace/Results/yolo_preds")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.predict(source=args.source, imgsz=args.imgsz, save=True, project=args.save_dir, name="preds")
    print(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())