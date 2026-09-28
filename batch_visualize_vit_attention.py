from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
from PIL import Image

import train_classifier as trainer
import visualize_vit_attention as viz


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Batch-generate ViT attention maps from a dataset split.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=project_dir / "outputs" / "vit_small_eurosat_40ep_pretrained" / "best_model.pt",
        help="Path to a trained ViT checkpoint.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir.parent / "data" / "EuroSAT_RGB" / "EuroSAT_RGB",
        help="Path to the EuroSAT RGB class-folder dataset.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional split manifest. Defaults to the checkpoint output directory manifest.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split to sample from.",
    )
    parser.add_argument(
        "--num-correct",
        type=int,
        default=4,
        help="Number of correctly classified examples to visualize.",
    )
    parser.add_argument(
        "--num-incorrect",
        type=int,
        default=4,
        help="Number of misclassified examples to visualize.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "vit_attention_batch",
        help="Directory where attention-map images and CSV metadata will be saved.",
    )
    return parser.parse_args()


def load_split_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise ValueError(f"No rows found for split '{split}' in {manifest_path}")
    return rows


def score_rows(
    model: torch.nn.Module,
    rows: list[dict[str, str]],
    data_dir: Path,
    class_names: list[str],
    image_size: int,
    device: torch.device,
) -> list[dict[str, object]]:
    _, eval_transform = trainer.build_transforms(image_size)
    scored: list[dict[str, object]] = []

    for row in rows:
        image_path = data_dir / row["relative_path"]
        image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
        input_tensor = eval_transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()

        predicted_index = int(probabilities.argmax().item())
        predicted_label = class_names[predicted_index]
        confidence = float(probabilities[predicted_index].item())
        true_label = row["class_name"]

        scored.append(
            {
                "relative_path": row["relative_path"],
                "true_label": true_label,
                "predicted_label": predicted_label,
                "predicted_index": predicted_index,
                "confidence": confidence,
                "correct": predicted_label == true_label,
            }
        )

    return scored


def select_examples(
    scored_rows: list[dict[str, object]],
    num_correct: int,
    num_incorrect: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = random.Random(seed)
    correct = [row for row in scored_rows if row["correct"]]
    incorrect = [row for row in scored_rows if not row["correct"]]

    rng.shuffle(correct)
    rng.shuffle(incorrect)

    selected = correct[:num_correct] + incorrect[:num_incorrect]
    selected.sort(key=lambda row: (not bool(row["correct"]), str(row["relative_path"])))
    return selected


def generate_attention_maps(
    model: torch.nn.Module,
    scored_row: dict[str, object],
    data_dir: Path,
    image_size: int,
    output_dir: Path,
) -> dict[str, object]:
    _, eval_transform = trainer.build_transforms(image_size)
    image_path = data_dir / str(scored_row["relative_path"])
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    input_tensor = eval_transform(image).unsqueeze(0).to(next(model.parameters()).device)

    attn_inputs, handles = viz.capture_attention_inputs(model)
    with torch.no_grad():
        logits = model(input_tensor)
    for handle in handles:
        handle.remove()

    attention_maps = viz.extract_attention_maps(model, attn_inputs)
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
    predicted_index = int(probabilities.argmax().item())
    predicted_probability = float(probabilities[predicted_index].item())

    last_layer_map = viz.cls_attention_to_grid(attention_maps[-1], image_size=image_size)
    rollout_map = viz.attention_rollout(attention_maps, image_size=image_size)

    image_stem = Path(str(scored_row["relative_path"])).with_suffix("").name
    suffix = "correct" if scored_row["correct"] else "incorrect"
    output_path = output_dir / f"{image_stem}_{suffix}_attention.png"

    viz.save_visualization(
        image=image,
        relative_path=str(scored_row["relative_path"]),
        true_label=str(scored_row["true_label"]),
        predicted_label=str(scored_row["predicted_label"]),
        predicted_probability=predicted_probability,
        last_layer_map=last_layer_map,
        rollout_map=rollout_map,
        output_path=output_path,
    )

    return {
        **scored_row,
        "predicted_probability": predicted_probability,
        "output_path": str(output_path),
    }


def write_summary(output_dir: Path, rows: list[dict[str, object]]) -> None:
    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "relative_path",
        "true_label",
        "predicted_label",
        "confidence",
        "predicted_probability",
        "correct",
        "output_path",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["args"]["model"] != "vit_small":
        raise ValueError(f"Checkpoint model is '{checkpoint['args']['model']}', expected 'vit_small'.")

    model = trainer.build_model(
        model_name="vit_small",
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
        image_size=int(checkpoint["args"]["image_size"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    viz.disable_fused_attention(model)

    manifest_path = args.split_manifest or args.checkpoint.parent / "split_manifest.csv"
    split_rows = load_split_rows(manifest_path, args.split)
    scored_rows = score_rows(
        model=model,
        rows=split_rows,
        data_dir=args.data_dir,
        class_names=checkpoint["class_names"],
        image_size=int(checkpoint["args"]["image_size"]),
        device=device,
    )

    selected_rows = select_examples(
        scored_rows=scored_rows,
        num_correct=args.num_correct,
        num_incorrect=args.num_incorrect,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered_rows = []
    for row in selected_rows:
        rendered_rows.append(
            generate_attention_maps(
                model=model,
                scored_row=row,
                data_dir=args.data_dir,
                image_size=int(checkpoint["args"]["image_size"]),
                output_dir=args.output_dir,
            )
        )

    write_summary(args.output_dir, rendered_rows)

    print(f"Saved {len(rendered_rows)} attention-map visualizations to: {args.output_dir}")
    print(f"Summary CSV: {args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
