from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
import sys

REPORT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = REPORT_DIR.parent
OUTPUTS_DIR = PROJECT_DIR / "outputs"
DATA_DIR = REPORT_DIR / "data"

sys.path.insert(0, str(PROJECT_DIR))

from train_classifier import build_model

RUNS = [
    {
        "run_id": "resnet18_eurosat_40ep",
        "label": "ResNet18 Scratch",
        "architecture": "ResNet18",
        "family": "CNN",
        "pretrained": "No",
        "marker": "square*",
        "color": "resnet18scratch",
    },
    {
        "run_id": "resnet18_eurosat_40ep_pretrained",
        "label": "ResNet18 Pretrained",
        "architecture": "ResNet18",
        "family": "CNN",
        "pretrained": "Yes",
        "marker": "square*",
        "color": "resnet18pre",
    },
    {
        "run_id": "resnet50_eurosat_40ep_bs256",
        "label": "ResNet50 Scratch",
        "architecture": "ResNet50",
        "family": "CNN",
        "pretrained": "No",
        "marker": "triangle*",
        "color": "resnet50scratch",
    },
    {
        "run_id": "resnet50_eurosat_40ep_pretrained_bs256",
        "label": "ResNet50 Pretrained",
        "architecture": "ResNet50",
        "family": "CNN",
        "pretrained": "Yes",
        "marker": "triangle*",
        "color": "resnet50pre",
    },
    {
        "run_id": "vit_small_eurosat_40ep",
        "label": "ViT-Small Scratch",
        "architecture": "ViT-Small",
        "family": "Transformer",
        "pretrained": "No",
        "marker": "*",
        "color": "vitscratch",
    },
    {
        "run_id": "vit_small_eurosat_40ep_pretrained",
        "label": "ViT-Small Pretrained",
        "architecture": "ViT-Small",
        "family": "Transformer",
        "pretrained": "Yes",
        "marker": "*",
        "color": "vitpre",
    },
]

ARCHITECTURE_TO_MODEL = {
    "ResNet18": "resnet18",
    "ResNet50": "resnet50",
    "ViT-Small": "vit_small",
}

ATTENTION_SOURCE = OUTPUTS_DIR / "vit_attention_batch" / "summary.csv"


def percent(value: float) -> float:
    return 100.0 * value


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_metrics(run_id: str) -> dict[str, object]:
    metrics_path = OUTPUTS_DIR / run_id / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def count_parameters() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for architecture, model_key in ARCHITECTURE_TO_MODEL.items():
        model = build_model(model_key, num_classes=10, pretrained=False, image_size=64)
        params = sum(param.numel() for param in model.parameters())
        counts[(architecture, "No")] = params
        counts[(architecture, "Yes")] = params
    return counts


def write_training_history() -> None:
    for run in RUNS:
        metrics = load_metrics(run["run_id"])
        rows = []
        for entry in metrics["history"]:
            rows.append(
                {
                    "epoch": entry["epoch"],
                    "train_loss": entry["train_loss"],
                    "train_accuracy_pct": percent(entry["train_accuracy"]),
                    "val_loss": entry["val_loss"],
                    "val_accuracy_pct": percent(entry["val_accuracy"]),
                    "learning_rate": entry["learning_rate"],
                }
            )
        write_csv(
            DATA_DIR / f"history_{run['run_id']}.csv",
            ["epoch", "train_loss", "train_accuracy_pct", "val_loss", "val_accuracy_pct", "learning_rate"],
            rows,
        )


def write_summary_tables(parameter_counts: dict[tuple[str, str], int]) -> None:
    summary_rows: list[dict[str, object]] = []
    extremes_rows: list[dict[str, object]] = []
    scatter_rows: list[dict[str, object]] = []

    for run in RUNS:
        metrics = load_metrics(run["run_id"])
        report = metrics["classification_report"]
        classes = {
            key: value
            for key, value in report.items()
            if isinstance(value, dict) and "f1-score" in value and key not in {"macro avg", "weighted avg"}
        }
        easiest_class, easiest_values = max(classes.items(), key=lambda item: item[1]["f1-score"])
        hardest_class, hardest_values = min(classes.items(), key=lambda item: item[1]["f1-score"])
        best_epoch = max(metrics["history"], key=lambda item: item["val_accuracy"])["epoch"]
        params = parameter_counts[(run["architecture"], run["pretrained"])]

        summary_rows.append(
            {
                "label": run["label"],
                "architecture": run["architecture"],
                "family": run["family"],
                "pretrained": run["pretrained"],
                "params_m": round(params / 1_000_000, 2),
                "best_val_acc_pct": round(percent(metrics["best_val_accuracy"]), 2),
                "test_acc_pct": round(percent(metrics["test_accuracy"]), 2),
                "macro_f1_pct": round(percent(report["macro avg"]["f1-score"]), 2),
                "test_loss": round(metrics["test_loss"], 4),
                "train_seconds": round(metrics["train_seconds_total"], 2),
                "best_epoch": best_epoch,
            }
        )
        extremes_rows.append(
            {
                "label": run["label"],
                "hardest_class": hardest_class,
                "hardest_f1_pct": round(percent(hardest_values["f1-score"]), 2),
                "easiest_class": easiest_class,
                "easiest_f1_pct": round(percent(easiest_values["f1-score"]), 2),
            }
        )
        scatter_rows.append(
            {
                "label": run["label"],
                "params_m": round(params / 1_000_000, 2),
                "test_acc_pct": round(percent(metrics["test_accuracy"]), 2),
                "family": run["family"],
                "pretrained": run["pretrained"],
            }
        )

    write_csv(
        DATA_DIR / "run_summary.csv",
        [
            "label",
            "architecture",
            "family",
            "pretrained",
            "params_m",
            "best_val_acc_pct",
            "test_acc_pct",
            "macro_f1_pct",
            "test_loss",
            "train_seconds",
            "best_epoch",
        ],
        summary_rows,
    )
    write_csv(
        DATA_DIR / "class_extremes.csv",
        ["label", "hardest_class", "hardest_f1_pct", "easiest_class", "easiest_f1_pct"],
        extremes_rows,
    )
    write_csv(
        DATA_DIR / "accuracy_vs_params.csv",
        ["label", "params_m", "test_acc_pct", "family", "pretrained"],
        scatter_rows,
    )


def write_pretraining_gain() -> None:
    pairs = [
        ("ResNet18", "resnet18_eurosat_40ep", "resnet18_eurosat_40ep_pretrained"),
        ("ResNet50", "resnet50_eurosat_40ep_bs256", "resnet50_eurosat_40ep_pretrained_bs256"),
        ("ViT-Small", "vit_small_eurosat_40ep", "vit_small_eurosat_40ep_pretrained"),
    ]
    rows = []
    for architecture, scratch_run, pretrained_run in pairs:
        scratch_metrics = load_metrics(scratch_run)
        pretrained_metrics = load_metrics(pretrained_run)
        scratch_acc = percent(scratch_metrics["test_accuracy"])
        pretrained_acc = percent(pretrained_metrics["test_accuracy"])
        rows.append(
            {
                "architecture": architecture,
                "scratch_test_acc_pct": round(scratch_acc, 2),
                "pretrained_test_acc_pct": round(pretrained_acc, 2),
                "gain_pct_points": round(pretrained_acc - scratch_acc, 2),
            }
        )
    write_csv(
        DATA_DIR / "pretraining_gain.csv",
        ["architecture", "scratch_test_acc_pct", "pretrained_test_acc_pct", "gain_pct_points"],
        rows,
    )


def write_confusion_matrix() -> None:
    best_run = "resnet50_eurosat_40ep_pretrained_bs256"
    metrics = load_metrics(best_run)
    class_names = metrics["class_names"]
    rows = []
    for actual_idx, row in enumerate(metrics["confusion_matrix"]):
        for predicted_idx, value in enumerate(row):
            rows.append(
                {
                    "actual_index": actual_idx + 1,
                    "predicted_index": predicted_idx + 1,
                    "actual_label": class_names[actual_idx],
                    "predicted_label": class_names[predicted_idx],
                    "count": value,
                }
            )
    write_csv(
        DATA_DIR / "confusion_best_model.csv",
        ["actual_index", "predicted_index", "actual_label", "predicted_label", "count"],
        rows,
    )


def write_top_errors() -> None:
    rows: list[dict[str, object]] = []
    for run in RUNS:
        metrics = load_metrics(run["run_id"])
        class_names = metrics["class_names"]
        errors = []
        for actual_idx, row in enumerate(metrics["confusion_matrix"]):
            for predicted_idx, value in enumerate(row):
                if actual_idx != predicted_idx and value:
                    errors.append((value, class_names[actual_idx], class_names[predicted_idx]))
        errors.sort(reverse=True)
        for rank, (count, actual, predicted) in enumerate(errors[:3], start=1):
            rows.append(
                {
                    "label": run["label"],
                    "rank": rank,
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                }
            )
    write_csv(DATA_DIR / "top_errors.csv", ["label", "rank", "actual", "predicted", "count"], rows)


def write_dataset_tables() -> None:
    manifest_path = OUTPUTS_DIR / "resnet18_eurosat_40ep" / "split_manifest.csv"
    split_counts = Counter()
    class_split_counts: Counter[tuple[str, str]] = Counter()
    total_counts = Counter()

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row["split"]
            class_name = row["class_name"]
            split_counts[split] += 1
            class_split_counts[(class_name, split)] += 1
            total_counts[class_name] += 1

    summary_rows = [
        {"split": "train", "count": split_counts["train"]},
        {"split": "validation", "count": split_counts["val"]},
        {"split": "test", "count": split_counts["test"]},
        {"split": "total", "count": sum(split_counts.values())},
    ]
    class_rows = []
    for class_name in sorted(total_counts):
        class_rows.append(
            {
                "class_name": class_name,
                "train_count": class_split_counts[(class_name, "train")],
                "val_count": class_split_counts[(class_name, "val")],
                "test_count": class_split_counts[(class_name, "test")],
                "total_count": total_counts[class_name],
            }
        )

    write_csv(DATA_DIR / "dataset_split_summary.csv", ["split", "count"], summary_rows)
    write_csv(
        DATA_DIR / "dataset_class_distribution.csv",
        ["class_name", "train_count", "val_count", "test_count", "total_count"],
        class_rows,
    )


def write_attention_summary() -> None:
    rows = []
    with ATTENTION_SOURCE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "relative_path": row["relative_path"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "confidence_pct": round(percent(float(row["confidence"])), 2),
                    "correct": row["correct"],
                }
            )
    write_csv(
        DATA_DIR / "attention_summary.csv",
        ["relative_path", "true_label", "predicted_label", "confidence_pct", "correct"],
        rows,
    )


def main() -> None:
    ensure_data_dir()
    parameter_counts = count_parameters()
    write_training_history()
    write_summary_tables(parameter_counts)
    write_pretraining_gain()
    write_confusion_matrix()
    write_top_errors()
    write_dataset_tables()
    write_attention_summary()


if __name__ == "__main__":
    main()
