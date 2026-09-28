# EuroSAT Results Summary

This file consolidates the final results from the six completed EuroSAT RGB training runs and the saved ViT attention-map batch.

## Overall Run Comparison

| Run | Pretrained | Best Val Acc. | Test Acc. | Macro F1 | Test Loss | Train Time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `resnet18_eurosat_40ep` | No | 94.96% | 95.75% | 95.62% | 0.1362 | 34.52 |
| `resnet18_eurosat_40ep_pretrained` | Yes | 98.12% | 98.42% | 98.36% | 0.0516 | 34.65 |
| `resnet50_eurosat_40ep_bs256` | No | 92.07% | 93.16% | 92.91% | 0.2202 | 76.02 |
| `resnet50_eurosat_40ep_pretrained_bs256` | Yes | 98.72% | 98.67% | 98.62% | 0.0458 | 75.29 |
| `vit_small_eurosat_40ep` | No | 89.68% | 90.40% | 90.08% | 0.3038 | 58.17 |
| `vit_small_eurosat_40ep_pretrained` | Yes | 98.49% | 98.54% | 98.48% | 0.0520 | 57.54 |

## Main Findings

- The best overall model was `resnet50_eurosat_40ep_pretrained_bs256` with 98.67% test accuracy, 98.62% macro F1, and the lowest test loss at 0.0458.
- The pretrained models consistently outperformed the from-scratch versions.
- Test-accuracy gain from pretraining was +2.67 percentage points for ResNet18, +5.51 points for ResNet50, and +8.15 points for ViT-Small.
- Among the non-pretrained runs, `resnet18_eurosat_40ep` was the strongest baseline at 95.75% test accuracy.
- `SeaLake` was the easiest class overall, reaching the top per-class F1 in five of the six runs.
- The hardest class was usually `PermanentCrop`; it was the lowest-F1 class for ResNet18, pretrained ResNet18, non-pretrained ResNet50, and pretrained ViT-Small. The weakest single class/result was `Highway` for non-pretrained ViT-Small at 78.84% F1.

## Per-Run Hardest and Easiest Classes

| Run | Lowest F1 Class | Lowest F1 | Highest F1 Class | Highest F1 |
| --- | --- | ---: | --- | ---: |
| `resnet18_eurosat_40ep` | `PermanentCrop` | 91.13% | `SeaLake` | 99.11% |
| `resnet18_eurosat_40ep_pretrained` | `PermanentCrop` | 96.39% | `SeaLake` | 99.67% |
| `resnet50_eurosat_40ep_bs256` | `PermanentCrop` | 86.83% | `SeaLake` | 99.00% |
| `resnet50_eurosat_40ep_pretrained_bs256` | `AnnualCrop` | 97.14% | `SeaLake` | 99.78% |
| `vit_small_eurosat_40ep` | `Highway` | 78.84% | `Forest` | 96.16% |
| `vit_small_eurosat_40ep_pretrained` | `PermanentCrop` | 96.61% | `SeaLake` | 99.78% |

## ViT Attention Batch

The batch visualization output under `outputs/vit_attention_batch/` contains 8 examples:

- 4 correct predictions and 4 incorrect predictions.
- The incorrect cases were `Forest -> HerbaceousVegetation` once, `PermanentCrop -> HerbaceousVegetation` twice, and `Residential -> Industrial` once.
- The repeated `PermanentCrop -> HerbaceousVegetation` confusion matches the broader classification trend where `PermanentCrop` is one of the more difficult classes.
- The correct examples were all predicted with very high confidence, between 99.95% and 100.00%.
- The incorrect examples were also often high-confidence, especially the three errors into `HerbaceousVegetation`, which indicates confident feature overlap rather than low-confidence ambiguity.

## Bottom Line

If one model needs to be reported as the strongest final result, use `resnet50_eurosat_40ep_pretrained_bs256`. If the report needs a smaller model with nearly identical accuracy, `vit_small_eurosat_40ep_pretrained` is close behind at 98.54% test accuracy.
