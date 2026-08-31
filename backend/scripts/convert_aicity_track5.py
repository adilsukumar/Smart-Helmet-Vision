from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = [
    "motorbike",
    "DHelmet",
    "DNoHelmet",
    "P1Helmet",
    "P1NoHelmet",
    "P2Helmet",
    "P2NoHelmet",
    "P0Helmet",
    "P0NoHelmet",
]


def build_image_index(frames_root: Path) -> dict[tuple[int, int], Path]:
    index: dict[tuple[int, int], Path] = {}
    for path in frames_root.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            video_id = int(path.parent.name)
            frame_id = int(path.stem)
        except ValueError:
            continue
        index[(video_id, frame_id)] = path
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AI City Track 5 boxes to YOLO format")
    parser.add_argument("--frames-root", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    grouped: dict[tuple[int, int], list[tuple[float, float, float, float, int]]] = defaultdict(list)
    with args.annotations.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or row[0].strip().startswith("#"):
                continue
            video, frame, left, top, width, height, class_id = map(float, row[:7])
            grouped[(int(video), int(frame))].append((left, top, width, height, int(class_id) - 1))

    videos = sorted({video for video, _ in grouped})
    random.Random(args.seed).shuffle(videos)
    val_count = max(1, round(len(videos) * args.val_ratio))
    val_videos = set(videos[:val_count])
    image_index = build_image_index(args.frames_root)
    missing: list[tuple[int, int]] = []

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    for (video_id, frame_id), boxes in sorted(grouped.items()):
        source = image_index.get((video_id, frame_id))
        if source is None:
            missing.append((video_id, frame_id))
            continue
        split = "val" if video_id in val_videos else "train"
        name = f"v{video_id:03d}_f{frame_id:06d}"
        destination = args.out / "images" / split / f"{name}{source.suffix.lower()}"
        shutil.copy2(source, destination)

        lines = []
        for left, top, width, height, class_index in boxes:
            if not 0 <= class_index < len(CLASS_NAMES):
                raise ValueError(f"Unexpected class id {class_index + 1}")
            center_x = (left + width / 2.0) / args.width
            center_y = (top + height / 2.0) / args.height
            lines.append(f"{class_index} {center_x:.8f} {center_y:.8f} {width / args.width:.8f} {height / args.height:.8f}")
        (args.out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset_yaml = "\n".join(
        [
            f"path: {args.out.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    (args.out / "dataset.yaml").write_text(dataset_yaml, encoding="utf-8")
    print(f"Converted {len(grouped) - len(missing)} annotated frames. Missing images: {len(missing)}")
    if missing:
        print("First missing keys:", missing[:10])
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

