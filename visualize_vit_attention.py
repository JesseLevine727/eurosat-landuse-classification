from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import train_classifier as trainer


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Visualize ViT attention maps for a EuroSAT sample.")
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
        help="Optional split manifest to choose a sample from.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to sample from when using the manifest.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Index within the chosen split manifest subset.",
    )
    parser.add_argument(
        "--relative-path",
        type=str,
        default=None,
        help="Relative dataset path, for example 'AnnualCrop/AnnualCrop_1002.jpg'. Overrides split selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "outputs" / "vit_attention_maps",
        help="Directory where the visualization image will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device used for inference.",
    )
    return parser.parse_args()


def resolve_manifest_path(args: argparse.Namespace) -> Path:
    if args.split_manifest is not None:
        return args.split_manifest
    return args.checkpoint.parent / "split_manifest.csv"


def select_relative_path(args: argparse.Namespace) -> tuple[str, str]:
    if args.relative_path is not None:
        class_name = Path(args.relative_path).parts[0]
        return args.relative_path, class_name

    manifest_path = resolve_manifest_path(args)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split]

    if not rows:
        raise ValueError(f"No rows found for split '{args.split}' in {manifest_path}")
    if args.sample_index < 0 or args.sample_index >= len(rows):
        raise IndexError(f"sample-index {args.sample_index} is out of range for split '{args.split}'")

    row = rows[args.sample_index]
    return row["relative_path"], row["class_name"]


def disable_fused_attention(model: torch.nn.Module) -> None:
    for block in model.blocks:
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False


def capture_attention_inputs(model: torch.nn.Module) -> tuple[list[torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    attn_inputs: list[torch.Tensor] = []
    handles = []

    def hook(_: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        attn_inputs.append(inputs[0].detach().clone())

    for block in model.blocks:
        handles.append(block.attn.register_forward_pre_hook(hook))

    return attn_inputs, handles


def extract_attention_maps(model: torch.nn.Module, attn_inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    attn_maps: list[torch.Tensor] = []
    for block, x in zip(model.blocks, attn_inputs):
        attn = block.attn
        batch_size, num_tokens, _ = x.shape
        qkv = attn.qkv(x).reshape(batch_size, num_tokens, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        q, k = attn.q_norm(q), attn.k_norm(k)
        scores = (q * attn.scale) @ k.transpose(-2, -1)
        probs = scores.softmax(dim=-1)
        attn_maps.append(probs.detach().cpu())
    return attn_maps


def cls_attention_to_grid(attn_map: torch.Tensor, image_size: int) -> np.ndarray:
    cls_to_patches = attn_map.mean(dim=1)[0, 0, 1:]
    patches_per_side = int(cls_to_patches.numel() ** 0.5)
    grid = cls_to_patches.reshape(1, 1, patches_per_side, patches_per_side)
    upsampled = F.interpolate(grid, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return upsampled.squeeze().numpy()


def attention_rollout(attn_maps: list[torch.Tensor], image_size: int) -> np.ndarray:
    num_tokens = attn_maps[0].shape[-1]
    joint = torch.eye(num_tokens)

    for attn_map in attn_maps:
        attn = attn_map.mean(dim=1)[0]
        attn = attn + torch.eye(num_tokens)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        joint = attn @ joint

    cls_to_patches = joint[0, 1:]
    patches_per_side = int(cls_to_patches.numel() ** 0.5)
    grid = cls_to_patches.reshape(1, 1, patches_per_side, patches_per_side)
    upsampled = F.interpolate(grid, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return upsampled.squeeze().numpy()


def normalize_map(attn_map: np.ndarray) -> np.ndarray:
    min_value = float(attn_map.min())
    max_value = float(attn_map.max())
    if max_value - min_value < 1e-8:
        return np.zeros_like(attn_map)
    return (attn_map - min_value) / (max_value - min_value)


def save_visualization(
    image: Image.Image,
    relative_path: str,
    true_label: str,
    predicted_label: str,
    predicted_probability: float,
    last_layer_map: np.ndarray,
    rollout_map: np.ndarray,
    output_path: Path,
) -> None:
    image_array = np.asarray(image)
    last_layer_map = normalize_map(last_layer_map)
    rollout_map = normalize_map(rollout_map)

    figure, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_array)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(last_layer_map, cmap="inferno")
    axes[1].set_title("Last-Layer CLS")
    axes[1].axis("off")

    axes[2].imshow(rollout_map, cmap="inferno")
    axes[2].set_title("Attention Rollout")
    axes[2].axis("off")

    axes[3].imshow(image_array)
    axes[3].imshow(rollout_map, cmap="inferno", alpha=0.45)
    axes[3].set_title("Rollout Overlay")
    axes[3].axis("off")

    figure.suptitle(
        f"{relative_path}\ntrue={true_label} | pred={predicted_label} ({predicted_probability:.3f})",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    checkpoint_args = checkpoint["args"]
    if checkpoint_args["model"] != "vit_small":
        raise ValueError(f"Checkpoint model is '{checkpoint_args['model']}', expected 'vit_small'.")

    image_size = int(checkpoint_args["image_size"])
    model = trainer.build_model(
        model_name="vit_small",
        num_classes=len(checkpoint["class_names"]),
        pretrained=False,
        image_size=image_size,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    disable_fused_attention(model)

    relative_path, true_label = select_relative_path(args)
    image_path = args.data_dir / relative_path
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size))

    _, eval_transform = trainer.build_transforms(image_size)
    input_tensor = eval_transform(image).unsqueeze(0).to(device)

    attn_inputs, handles = capture_attention_inputs(model)
    with torch.no_grad():
        logits = model(input_tensor)
    for handle in handles:
        handle.remove()

    attention_maps = extract_attention_maps(model, attn_inputs)
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu()
    predicted_index = int(probabilities.argmax().item())
    predicted_label = checkpoint["class_names"][predicted_index]
    predicted_probability = float(probabilities[predicted_index].item())

    last_layer_map = cls_attention_to_grid(attention_maps[-1], image_size=image_size)
    rollout_map = attention_rollout(attention_maps, image_size=image_size)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_name = Path(relative_path).with_suffix("").name + "_attention.png"
    output_path = args.output_dir / output_name
    save_visualization(
        image=image,
        relative_path=relative_path,
        true_label=true_label,
        predicted_label=predicted_label,
        predicted_probability=predicted_probability,
        last_layer_map=last_layer_map,
        rollout_map=rollout_map,
        output_path=output_path,
    )

    print(f"Saved attention visualization to: {output_path}")
    print(f"True label: {true_label}")
    print(f"Predicted label: {predicted_label}")
    print(f"Predicted probability: {predicted_probability:.4f}")


if __name__ == "__main__":
    main()
