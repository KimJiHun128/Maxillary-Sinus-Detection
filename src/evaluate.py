"""Evaluate a trained detector on the held-out test split."""

import argparse

import numpy as np
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/sinus_ct.yaml")
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = YOLO(args.weights).val(
        data=args.data,
        split="test",
        imgsz=512,
        max_det=2,
        device=args.device,
    )
    print(f"precision: {metrics.box.mp:.4f}")
    print(f"recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mean F1: {np.mean(metrics.box.f1):.4f}")


if __name__ == "__main__":
    main()

