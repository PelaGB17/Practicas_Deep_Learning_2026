import argparse
import csv
import json
from pathlib import Path
from typing import List

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from final_project.data import build_eval_transform, list_class_names_from_directory
from final_project.modeling import build_resnet50, resolve_device
from final_project.settings import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_REPORTS_DIR,
    DEFAULT_VAL_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on validation data.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: Path):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint format not supported. Expected key 'model_state_dict'. "
            "Train with python -m final_project.train first."
        )
    return checkpoint


def write_confusion_matrix_csv(output_path: Path, matrix, class_names: List[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + class_names)
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + list(row))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.checkpoint)
    class_names = checkpoint.get("class_names")
    if not class_names:
        class_names = list_class_names_from_directory(args.val_dir)

    image_size = int(checkpoint.get("image_size", args.image_size))
    mean = tuple(checkpoint.get("mean", IMAGENET_MEAN))
    std = tuple(checkpoint.get("std", IMAGENET_STD))

    val_dataset = datasets.ImageFolder(
        args.val_dir,
        transform=build_eval_transform(image_size=image_size, mean=mean, std=std),
    )
    if val_dataset.classes != class_names:
        raise ValueError(
            "Class names from checkpoint do not match validation folder classes."
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_resnet50(num_classes=len(class_names), unfreeze_blocks=0, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    for images, labels in val_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        preds = logits.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    report_path = args.output_dir / "classification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    matrix_path = args.output_dir / "confusion_matrix.csv"
    write_confusion_matrix_csv(matrix_path, matrix, class_names)

    summary = {
        "checkpoint": str(args.checkpoint),
        "num_samples": len(y_true),
        "accuracy": report.get("accuracy", 0.0),
        "macro_precision": report.get("macro avg", {}).get("precision", 0.0),
        "macro_recall": report.get("macro avg", {}).get("recall", 0.0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
        "report_path": str(report_path),
        "confusion_matrix_path": str(matrix_path),
    }
    summary_path = args.output_dir / "summary_metrics.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
