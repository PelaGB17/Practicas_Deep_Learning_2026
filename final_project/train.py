import argparse
import copy
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from final_project.data import create_dataloaders
from final_project.modeling import build_resnet50, resolve_device
from final_project.settings import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_TRAIN_DIR,
    DEFAULT_VAL_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transfer learning model for final project.")
    parser.add_argument("--train-dir", type=Path, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--output-checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="mlii-final-project")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-run-name", type=str, default="")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def write_history_csv(history: List[Dict[str, float]], output_path: Path) -> None:
    fieldnames = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def maybe_init_wandb(args: argparse.Namespace) -> Optional[Any]:
    if not args.use_wandb:
        return None

    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "wandb is not installed. Install final_project/requirements.txt before using --use-wandb."
        ) from exc

    api_key = os.getenv("WANDB_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "WANDB_API_KEY is not set. Export it in your shell before running training with --use-wandb."
        )

    wandb.login(key=api_key)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or None,
        config=vars(args),
    )
    return run


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if not args.train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {args.train_dir}")
    if not args.val_dir.exists():
        raise FileNotFoundError(f"Validation directory not found: {args.val_dir}")

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    train_loader, val_loader, class_names = create_dataloaders(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    model = build_resnet50(
        num_classes=len(class_names),
        unfreeze_blocks=args.unfreeze_blocks,
        pretrained=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    wandb_run: Optional[Any] = maybe_init_wandb(args)

    best_epoch = 0
    best_val_accuracy = -1.0
    best_state_dict = copy.deepcopy(model.state_dict())
    history: List[Dict[str, float]] = []

    print(f"Device: {device}")
    print(f"Classes ({len(class_names)}): {class_names}")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} | "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={row['val_accuracy']:.4f}"
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        if wandb_run is not None:
            wandb_run.log(row, step=epoch)

    checkpoint = {
        "model_state_dict": best_state_dict,
        "class_names": class_names,
        "num_classes": len(class_names),
        "image_size": args.image_size,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "backbone": "resnet50",
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_args": vars(args),
    }

    torch.save(checkpoint, args.output_checkpoint)

    history_path = args.output_checkpoint.with_suffix(".history.csv")
    write_history_csv(history, history_path)

    summary = {
        "checkpoint": str(args.output_checkpoint),
        "history_csv": str(history_path),
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "num_classes": len(class_names),
    }
    summary_path = args.output_checkpoint.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved checkpoint:", args.output_checkpoint)
    print("Saved history:", history_path)
    print("Saved summary:", summary_path)

    if wandb_run is not None:
        wandb_run.summary.update(summary)
        wandb_run.finish()


if __name__ == "__main__":
    main()
