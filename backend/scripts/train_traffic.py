from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a small YOLO traffic detector")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--export", choices=["none", "ncnn", "onnx"], default="none")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install requirements-ml.txt before training") from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs/traffic",
        name="helmet_rules",
        seed=42,
    )
    metrics = model.val(data=args.data)
    print(metrics)
    if args.export != "none":
        model.export(format=args.export, imgsz=args.imgsz)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

