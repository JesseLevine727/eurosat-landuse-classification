from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import timm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MODEL_SPECS = {
    "resnet18": {"family": "torchvision", "builder": models.resnet18, "weights": models.ResNet18_Weights.DEFAULT},
    "resnet50": {"family": "torchvision", "builder": models.resnet50, "weights": models.ResNet50_Weights.DEFAULT},
    "vit_small": {"family": "timm", "name": "vit_small_patch16_224"},
}


def namespace_to_jsonable_dict(args: argparse.Namespace) -> dict[str, object]:
    jsonable: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            jsonable[key] = str(value)
        else:
            jsonable[key] = value
    return jsonable


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    repo_root = project_dir.parent

    parser = argparse.ArgumentParser(description="Train an image model on EuroSAT RGB.")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=sorted(MODEL_SPECS.keys()),
        help="Backbone to train.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=repo_root / "data" / "EuroSAT_RGB" / "EuroSAT_RGB",
        help="Path to the EuroSAT RGB class-folder dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, metrics, and plots.",
    )
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--image-size", type=int, default=64, help="Image size for training.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Start from ImageNet-pretrained weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Training device.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        suffix = "_pretrained" if args.pretrained else ""
        args.output_dir = project_dir / "outputs" / f"{args.model}_eurosat_{args.epochs}ep{suffix}"
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def build_splits(data_dir: Path, seed: int) -> tuple[datasets.ImageFolder, np.ndarray, np.ndarray, np.ndarray]:
    base_dataset = datasets.ImageFolder(root=str(data_dir))
    indices = np.arange(len(base_dataset))
    targets = np.array(base_dataset.targets)

    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=targets,
    )
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=seed,
        stratify=targets[temp_indices],
    )

    return base_dataset, train_indices, val_indices, test_indices


def write_split_manifest(
    output_dir: Path,
    base_dataset: datasets.ImageFolder,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:
    split_lookup: dict[int, str] = {}
    for split_name, split_indices in (
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ):
        for idx in split_indices:
            split_lookup[int(idx)] = split_name

    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "class_index", "class_name", "split"])
        for idx, (path, class_index) in enumerate(base_dataset.samples):
            writer.writerow(
                [
                    str(Path(path).relative_to(base_dataset.root)),
                    class_index,
                    base_dataset.classes[class_index],
                    split_lookup[idx],
                ]
            )


def create_dataloaders(
    data_dir: Path,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    test_indices: np.ndarray,
    image_size: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_transform, eval_transform = build_transforms(image_size)

    train_dataset = datasets.ImageFolder(root=str(data_dir), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(data_dir), transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=str(data_dir), transform=eval_transform)

    train_loader = DataLoader(
        Subset(train_dataset, train_indices.tolist()),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader, test_loader, train_dataset.classes


def build_model(model_name: str, num_classes: int, pretrained: bool, image_size: int) -> nn.Module:
    spec = MODEL_SPECS[model_name]

    if spec["family"] == "torchvision":
        model_builder = spec["builder"]
        default_weights = spec["weights"]
        weights = default_weights if pretrained else None
        model = model_builder(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if spec["family"] == "timm":
        return timm.create_model(
            spec["name"],
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=image_size,
        )

    raise ValueError(f"Unsupported model family for {model_name}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None = None,
    scaler: GradScaler | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        if is_training:
            assert optimizer is not None
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

    return total_loss / total_examples, total_correct / total_examples


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        predictions = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    return (
        total_loss / total_examples,
        total_correct / total_examples,
        all_labels,
        all_predictions,
    )


def save_history_plot(history: list[dict[str, float]], output_dir: Path) -> None:
    epochs = [entry["epoch"] for entry in history]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, [entry["train_loss"] for entry in history], label="train")
    plt.plot(epochs, [entry["val_loss"] for entry in history], label="val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [entry["train_accuracy"] for entry in history], label="train")
    plt.plot(epochs, [entry["val_accuracy"] for entry in history], label="val")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")

    seed_everything(args.seed)
    configure_runtime(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_dataset, train_indices, val_indices, test_indices = build_splits(args.data_dir, args.seed)
    write_split_manifest(args.output_dir, base_dataset, train_indices, val_indices, test_indices)

    pin_memory = device.type == "cuda"
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=args.data_dir,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = build_model(
        model_name=args.model,
        num_classes=len(class_names),
        pretrained=args.pretrained,
        image_size=args.image_size,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(device.type, enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    best_val_accuracy = -1.0
    best_checkpoint_path = args.output_dir / "best_model.pt"
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_accuracy, _, _ = evaluate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": time.perf_counter() - epoch_start,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f} | "
            f"time={epoch_metrics['epoch_seconds']:.1f}s"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "args": namespace_to_jsonable_dict(args),
                    "history": history,
                },
                best_checkpoint_path,
            )

    checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy, test_labels, test_predictions = evaluate_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    summary = {
        "config": namespace_to_jsonable_dict(args),
        "device": str(device),
        "class_names": class_names,
        "split_counts": {
            "train": len(train_indices),
            "val": len(val_indices),
            "test": len(test_indices),
        },
        "best_val_accuracy": best_val_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "train_seconds_total": time.perf_counter() - train_start,
        "history": history,
        "classification_report": classification_report(
            test_labels,
            test_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(test_labels, test_predictions).tolist(),
    }

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    save_history_plot(history, args.output_dir)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
