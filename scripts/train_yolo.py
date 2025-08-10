#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train Ultralytics YOLO on the converted GPR dataset")
    p.add_argument("--data_yaml", type=str, default="/workspace/data/gpr_yolo/gpr.yaml", help="Path to dataset YAML")
    p.add_argument("--model", type=str, default="yolov8n.pt", help="Ultralytics model or config path")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=str, default="/workspace/Results/yolo_runs")
    p.add_argument("--name", type=str, default="gpr_yolov8")
    return p.parse_args()


def ensure_requirements():
    req = Path("/workspace/requirements.txt")
    if req.exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req), "--quiet"], check=False)


def main() -> int:
    args = parse_args()
    ensure_requirements()
    try:
        from ultralytics import YOLO
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics>=8.0.0"])  # fallback
        from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
    )
    print(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())