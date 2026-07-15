"""Train the YOLOv8 maxillary-sinus detector described in the paper."""

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/sinus_ct.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--project", default="runs/sinus_ct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_source = "yolov8n.yaml" if args.from_scratch else "yolov8n.pt"
    run_name = "yolov8n_scratch" if args.from_scratch else "yolov8n_transfer"
    model = YOLO(model_source)
    model.train(
        data=args.data,
        epochs=300,
        patience=50,
        batch=32,
        imgsz=512,
        optimizer="AdamW",
        lr0=0.00125,
        momentum=0.9,
        device=args.device,
        project=args.project,
        name=run_name,
    )


if __name__ == "__main__":
    main()

