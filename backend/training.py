from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = [
    "motorbike", "DHelmet", "DNoHelmet", "P1Helmet", "P1NoHelmet",
    "P2Helmet", "P2NoHelmet", "P0Helmet", "P0NoHelmet",
]


def convert(args) -> int:
    images = {}
    for path in args.frames_root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            images[(int(path.parent.name), int(path.stem))] = path
        except ValueError:
            pass

    grouped = defaultdict(list)
    with args.annotations.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            video, frame, left, top, width, height, class_id = map(float, row[:7])
            grouped[(int(video), int(frame))].append((left, top, width, height, int(class_id) - 1))

    videos = sorted({video for video, _ in grouped})
    random.Random(args.seed).shuffle(videos)
    validation = set(videos[:max(1, round(len(videos) * args.val_ratio))])
    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    missing = []
    for (video_id, frame_id), boxes in sorted(grouped.items()):
        source = images.get((video_id, frame_id))
        if source is None:
            missing.append((video_id, frame_id))
            continue
        split = "val" if video_id in validation else "train"
        name = f"v{video_id:03d}_f{frame_id:06d}"
        shutil.copy2(source, args.out / "images" / split / f"{name}{source.suffix.lower()}")
        lines = []
        for left, top, width, height, class_index in boxes:
            if not 0 <= class_index < len(CLASS_NAMES):
                raise ValueError(f"Unexpected class id {class_index + 1}")
            x = (left + width / 2) / args.width
            y = (top + height / 2) / args.height
            lines.append(f"{class_index} {x:.8f} {y:.8f} {width / args.width:.8f} {height / args.height:.8f}")
        (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml = [f"path: {args.out.resolve().as_posix()}", "train: images/train", "val: images/val", "names:"]
    yaml.extend(f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES))
    (args.out / "dataset.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
    print(f"Converted {len(grouped) - len(missing)} frames. Missing: {len(missing)}")
    return 2 if missing else 0


def train(args) -> int:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install backend/requirements.txt before training") from exc
    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                device=args.device, project="runs/traffic", name="helmet_rules", seed=42)
    print(model.val(data=args.data))
    if args.export != "none":
        model.export(format=args.export, imgsz=args.imgsz)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Dataset conversion and model training")
    commands = parser.add_subparsers(dest="command", required=True)
    converter = commands.add_parser("convert")
    converter.add_argument("--frames-root", required=True, type=Path)
    converter.add_argument("--annotations", required=True, type=Path)
    converter.add_argument("--out", required=True, type=Path)
    converter.add_argument("--width", type=int, default=1920)
    converter.add_argument("--height", type=int, default=1080)
    converter.add_argument("--val-ratio", type=float, default=0.2)
    converter.add_argument("--seed", type=int, default=42)
    trainer = commands.add_parser("train")
    trainer.add_argument("--data", required=True)
    trainer.add_argument("--model", default="yolo11n.pt")
    trainer.add_argument("--epochs", type=int, default=50)
    trainer.add_argument("--imgsz", type=int, default=640)
    trainer.add_argument("--batch", type=int, default=16)
    trainer.add_argument("--device", default=None)
    trainer.add_argument("--export", choices=["none", "ncnn", "onnx"], default="none")
    args = parser.parse_args()
    return convert(args) if args.command == "convert" else train(args)


if __name__ == "__main__":
    raise SystemExit(main())

