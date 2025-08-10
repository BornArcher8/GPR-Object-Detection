#!/usr/bin/env python3
import argparse
import os
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kwargs: x  # fallback if tqdm not installed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert GPR VOC annotations to YOLO format and split train/val")
    parser.add_argument("--source_root", type=str, default="/workspace/data/gpr-data-classifier/hyperbola-classifier",
                        help="Path to the original dataset root that contains images/ and annotations/xmls/")
    parser.add_argument("--output_root", type=str, default="/workspace/data/gpr_yolo",
                        help="Output root for YOLO dataset (images/, labels/, and dataset YAML)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--dataset_name", type=str, default="gpr",
                        help="Dataset name used for YAML file naming")
    return parser.parse_args()


def read_label_map(label_map_path: Path) -> list:
    # TensorFlow Object Detection label_map.pbtxt parsing for names
    if not label_map_path.exists():
        # Default single class
        return ["hyperbola"]
    text = label_map_path.read_text()
    names = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line.split(":", 1)[1].strip().strip("'\"")
            names.append(name)
    # Ensure deterministic ordering by id if present
    if not names:
        names = ["hyperbola"]
    return names


def convert_voc_box_to_yolo(bbox: tuple, img_w: int, img_h: int) -> tuple:
    # bbox is (xmin, ymin, xmax, ymax) in absolute pixels (1-based VOC)
    xmin, ymin, xmax, ymax = bbox
    # Clamp and convert to 0-based pixel space
    xmin = max(0, xmin - 1)
    ymin = max(0, ymin - 1)
    xmax = min(img_w - 1, xmax - 1)
    ymax = min(img_h - 1, ymax - 1)
    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    # Normalize
    return (
        cx / img_w if img_w > 0 else 0.0,
        cy / img_h if img_h > 0 else 0.0,
        bw / img_w if img_w > 0 else 0.0,
        bh / img_h if img_h > 0 else 0.0,
    )


def parse_voc_xml(xml_path: Path) -> tuple:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bnd = obj.find("bndbox")
        xmin = int(float(bnd.find("xmin").text))
        ymin = int(float(bnd.find("ymin").text))
        xmax = int(float(bnd.find("xmax").text))
        ymax = int(float(bnd.find("ymax").text))
        objects.append((name, (xmin, ymin, xmax, ymax)))
    return img_w, img_h, objects


def ensure_dirs(path_list: list) -> None:
    for p in path_list:
        Path(p).mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(output_root: Path, dataset_name: str, names: list) -> Path:
    yaml_path = output_root / f"{dataset_name}.yaml"
    yaml_content = (
        f"path: {output_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(yaml_content)
    return yaml_path


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    source_root = Path(args.source_root)
    images_dir = source_root / "images"
    xml_dir = source_root / "annotations" / "xmls"
    label_map_path = source_root / "annotations" / "label_map.pbtxt"

    if not images_dir.exists() or not xml_dir.exists():
        print(f"ERROR: Expected images at {images_dir} and XMLs at {xml_dir}", file=sys.stderr)
        return 1

    class_names = read_label_map(label_map_path)
    name_to_id_zero_based = {name: idx for idx, name in enumerate(class_names)}

    output_root = Path(args.output_root)
    images_train = output_root / "images" / "train"
    images_val = output_root / "images" / "val"
    labels_train = output_root / "labels" / "train"
    labels_val = output_root / "labels" / "val"
    ensure_dirs([images_train, images_val, labels_train, labels_val])

    # Gather samples by xml files that have a corresponding image file
    xml_files = sorted([p for p in xml_dir.glob("*.xml")])
    samples = []
    for xml_path in xml_files:
        # Try to infer image filename. Many xmls share base name with image.
        base = xml_path.stem
        # images are jpg per repo; try multiple extensions
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = images_dir / f"{base}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            # Some images may have different names; fall back to reading filename field
            try:
                tree = ET.parse(str(xml_path))
                filename_node = tree.getroot().find("filename")
                if filename_node is not None:
                    candidate = images_dir / filename_node.text
                    if candidate.exists():
                        img_path = candidate
            except Exception:
                pass
        if img_path is None:
            continue
        samples.append((img_path, xml_path))

    if len(samples) == 0:
        print("ERROR: No image-xml pairs found.", file=sys.stderr)
        return 1

    # Split
    random.shuffle(samples)
    val_count = max(1, int(len(samples) * args.val_split))
    val_samples = set(samples[:val_count])

    def convert_and_copy(sample, subset: str):
        img_path, xml_path = sample
        try:
            img_w, img_h, objects = parse_voc_xml(xml_path)
        except Exception as e:
            print(f"WARN: skip {xml_path} due to parse error: {e}", file=sys.stderr)
            return
        # write label file
        label_lines = []
        for name, bbox in objects:
            if name not in name_to_id_zero_based:
                # unknown class, skip
                continue
            cls_id = name_to_id_zero_based[name]
            x, y, w, h = convert_voc_box_to_yolo(bbox, img_w, img_h)
            # Skip invalid or zero-size boxes
            if w <= 0 or h <= 0:
                continue
            label_lines.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        # Copy image and write labels only if there is at least 1 box
        if not label_lines:
            return
        if subset == "train":
            dst_img = images_train / img_path.name
            dst_lbl = labels_train / (img_path.stem + ".txt")
        else:
            dst_img = images_val / img_path.name
            dst_lbl = labels_val / (img_path.stem + ".txt")
        shutil.copy2(str(img_path), str(dst_img))
        dst_lbl.write_text("\n".join(label_lines) + "\n")

    for sample in tqdm(samples, desc="Converting to YOLO"):
        subset = "val" if sample in val_samples else "train"
        convert_and_copy(sample, subset)

    yaml_path = write_dataset_yaml(output_root, args.dataset_name, class_names)
    print(f"Done. Wrote dataset YAML at: {yaml_path}")
    print(f"Train images: {len(list(images_train.glob('*')))}; Val images: {len(list(images_val.glob('*')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())