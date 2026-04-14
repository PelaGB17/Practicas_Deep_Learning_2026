from pathlib import Path
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from final_project.settings import IMAGENET_MEAN, IMAGENET_STD


def list_class_names_from_directory(data_dir: Path) -> List[str]:
    data_dir = Path(data_dir)
    classes = sorted([item.name for item in data_dir.iterdir() if item.is_dir()])
    if not classes:
        raise ValueError(f"No class folders found in {data_dir}")
    return classes


def build_train_transform(image_size: int = 224):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    mean: Tuple[float, float, float] = IMAGENET_MEAN,
    std: Tuple[float, float, float] = IMAGENET_STD,
):
    resize_size = int(image_size * 1.14)
    return transforms.Compose(
        [
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def create_dataloaders(
    train_dir: Path,
    val_dir: Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, List[str]]:
    train_dataset = datasets.ImageFolder(train_dir, transform=build_train_transform(image_size))
    val_dataset = datasets.ImageFolder(val_dir, transform=build_eval_transform(image_size))

    if train_dataset.classes != val_dataset.classes:
        raise ValueError("Train and validation class folders do not match.")

    pin_memory = True
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, train_dataset.classes
