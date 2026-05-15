# Static-Post-Training-Quantization-Analysis-Project
## INT8 Calibration Size Study — ImageNet-1K

Empirical study investigating how **calibration set size** affects INT8 Post-Training Static Quantization (PTQ) accuracy, calibration error, model size, and inference latency on ImageNet-1K.

## Research Question

> *Is a small calibration set (n=10) sufficient for INT8 PTQ, or is a large one (n=5000) necessary?*

We answer this empirically with **90 experiments**: 3 models × 6 calibration sizes × 5 random seeds.

## Key Findings

- **ResNet-18 / ResNet-50:** Highly robust to PTQ. ΔAcc < 0.5% across all calibration sizes — even n=10 is sufficient.
- **MobileNetV2:** Catastrophic, non-monotonic degradation due to per-tensor quantization of depthwise convolutions. ΔAcc reaches ~35% and does *not* improve with more calibration data.
- **Implication:** Calibration size alone is insufficient for MobileNet-family architectures; per-channel quantization is required.

## Experimental Setup

| Component | Value |
|-----------|-------|
| Dataset | ImageNet-1K validation (5,000 images sampled) |
| Models | ResNet-18, ResNet-50, MobileNetV2 |
| Calibration sizes | {10, 50, 100, 500, 1000, 5000} |
| Seeds | {42, 123, 456, 789, 1024} |
| Quantization backend | `fbgemm` (per-tensor) |
| Metrics | Top-1 Acc, ΔAcc, ECE, Model Size (MB), Latency (ms) |
| Hardware | Google Colab (T4 / A100 GPU) |

**Total runs:** 3 × 6 × 5 = **90 experiments**

## Project Structure

```
imagenet_v2/
├── config.py                                # All experiment parameters
├── requirements.txt
├── README.md
├── src/
│   ├── data_utils.py                        # ImageNet HF streaming + calibration sampling
│   ├── model_utils.py                       # Model loading (quantization-ready variants)
│   ├── quantization.py                      # PTQ pipeline (fuse → prepare → calibrate → convert)
│   ├── metrics.py                           # Top-1 Acc, ECE, model size, latency
│   └── experiment_runner.py                 # Orchestrates 90 experiments with resume
├── notebooks/
│   ├── 01_fp32_baseline.ipynb               # FP32 baseline accuracy
│   ├── 02_quantization_experiments.ipynb    # 90-run PTQ experiments
│   └── 03_analysis_and_visualization.ipynb  # Figures, tables, statistical tests
├── results/aggregated/
│   ├── all_results.csv                      # All 90 runs (flat)
│   ├── summary.csv                          # mean ± std per (model, n)
│   ├── fp32_baselines.csv                   # FP32 Top-1 per model
│   ├── statistical_tests.csv                # Wilcoxon, Spearman
│   └── tables.xlsx                          # Multi-sheet Excel
└── figures/                                 # PNG + PDF
    ├── accuracy_drop_vs_calib_size.{png,pdf}
    ├── ece_vs_calib_size.{png,pdf}
    ├── latency_size_comparison.{png,pdf}
    └── saturation_analysis.{png,pdf}
```

## How to Run

### Prerequisites

1. **HuggingFace account with ImageNet-1K access**
   - Sign up at [huggingface.co](https://huggingface.co)
   - Visit the [imagenet-1k dataset page](https://huggingface.co/datasets/ILSVRC/imagenet-1k) and click **"Agree and access repository"**
   - Generate a read token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

2. **Google Colab account** (T4 GPU is free; A100 requires Colab Pro)

3. **Google Drive** with the project folder uploaded at `MyDrive/imagenet_v2/`

### Step 1 — FP32 Baseline (Notebook 01)

Open `notebooks/01_fp32_baseline.ipynb` in Colab.

```
Runtime → Change runtime type → T4 GPU
```

Run all cells. You will be prompted to mount Google Drive and log in with your HuggingFace token.

Expected output:
```
ResNet-18    : 69.06%
ResNet-50    : 76.13%
MobileNetV2  : 71.00%
```

Results are saved to `results/aggregated/fp32_baselines.csv`.

### Step 2 — Quantization Experiments (Notebook 02)

Open `notebooks/02_quantization_experiments.ipynb` in Colab.

The notebook is split into per-model cells. Each cell runs 30 experiments (6 sizes × 5 seeds) and saves results to Drive immediately after completion.

```
Cell 1: Setup            → mount Drive, install packages, HF login
Cell 2: Helper           → save_to_drive(), count_completed()
Cell 3: ResNet-18        → 30 runs 
Cell 4: ResNet-50        → 30 runs 
Cell 5: MobileNetV2      → 30 runs 
Cell 6: Aggregate        → produce summary.csv
Cell 7: Validation       → quick n=10 vs n=5000 sanity check
Cell 8: Final Save       → push everything to Drive
```

**Resume-safe:** `resume=True` skips already-completed runs. If Colab disconnects mid-run, just re-run the cell — it continues from where it stopped.

### Step 3 — Analysis & Figures (Notebook 03)

Open `notebooks/03_analysis_and_visualization.ipynb` in Colab.

Run all cells. Produces:
- 4 figures (PDF + PNG) in `figures/`
- Statistical tests (Wilcoxon, Spearman)
- Excel workbook (`tables.xlsx`)

## Local Run (Optional — Smoke Test)

To verify the pipeline locally without running all 90 experiments:

```bash
pip install -r requirements.txt

python -c "
from src.experiment_runner import run_single_experiment
result = run_single_experiment('resnet18', calib_size=10, seed=42)
print(result)
"
```

Requires a working PyTorch installation and HuggingFace `HF_TOKEN` environment variable.

## Note on EfficientNet-B0

EfficientNet-B0 was initially considered but excluded due to incompatibility with PyTorch's eager-mode static PTQ (SiLU activations and SqueezeExcitation block multiplications lack quantized CPU kernels). FX graph mode quantization would be required, which is outside the scope of this study.

## Dependencies

See `requirements.txt`. Core packages:

```
torch                # Colab pre-installed
torchvision          # Colab pre-installed
datasets>=2.14.0
huggingface_hub>=0.19.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
scipy>=1.11.0
openpyxl
```

## Reproducibility

All experiments use fixed random seeds: `{42, 123, 456, 789, 1024}`. Calibration sampling is deterministic via `_set_seeds()` in `data_utils.py`. The 5 seeds yield the mean ± std reported in figures and tables.
