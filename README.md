# CrossHL-VLM

CrossHL-VLM extends Cross Hyperspectral and LiDAR Attention Transformer with a frozen CLIP text encoder used only as semantic regularization. CLIP is not used as the classifier, and hyperspectral data is not converted to RGB.

The Cross-HL classification path remains unchanged:

1. Native HSI and LiDAR patches are encoded by the Cross-HL backbone.
2. The fused CLS feature is classified by the standard linear head with cross-entropy loss.
3. The same fused CLS feature is projected into the CLIP text space by a lightweight MLP.
4. A low-weight semantic loss aligns the projected feature with frozen text prototypes.

The strongest prompt setting is `spectral_lidar`, which describes both HSI spectral/material behavior and LiDAR structural/elevation behavior.

## Repository Layout

- `model/CrossHL_model.py` - Cross-HL model with an optional semantic projection head.
- `data.py` - Trento, Houston, and MUUFL dataset loading plus percentage and K-shot sampling.
- `prompts.py` - class names and prompt banks for `name`, `spectral`, and `spectral_lidar`.
- `train_vlm_ablation.py` - reproducible ablation runner.
- `analyze_runs.py` - aggregate metrics, paired baseline deltas, OA plots, and t-SNE.
- `notebooks/CrossHL_VLM_Showcase.ipynb` - notebook interface for dataset inspection, ablations, and plots.
- `research_positioning_report.tex` - research positioning report.
- `utils.py`, `logger.py` - compatibility helpers from the original workflow.

Datasets, virtual environments, checkpoints, and run outputs are intentionally excluded from version control.

## Environment

Activate the local environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Check CUDA:

```powershell
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
PY
```

The runner also prints the selected device at startup.

## Datasets

The loader expects these local folders:

```text
Trento11x11/
HoustonDataset/
MUUFL_Dataset/
```

Each folder should contain:

```text
HSI_Tr.mat
HSI_Te.mat
LIDAR_Tr.mat
LIDAR_Te.mat
TrLabel.mat
TeLabel.mat
```

These folders are ignored by Git.

## Ablation Modes

- `baseline` - Cross-HL classifier only.
- `name` - semantic regularization with class-name prompts.
- `spectral` - semantic regularization with HSI spectral/material prompts.
- `spectral_lidar` - semantic regularization with HSI spectral plus LiDAR structural prompts.
- `name_low_lambda`, `spectral_low_lambda`, `spectral_lidar_low_lambda` - lower semantic weight variants.
- `spectral_lidar_high_lambda` - stronger semantic-weight diagnostic.

All experiments in the same iteration share the same few-shot split, enabling paired comparison with the baseline.

## Recommended Runs

Use explicit CUDA with mixed precision and TF32:

```powershell
python train_vlm_ablation.py --device cuda --amp --allow-tf32 --dataset Houston --shots 20 --experiments baseline name spectral spectral_lidar --epochs 200 --iterations 10
```

For a safer semantic-regularization test, use centered prototypes, delayed semantic activation, and a lower semantic weight:

```powershell
python train_vlm_ablation.py --device cuda --amp --allow-tf32 --center-prototypes --semantic-start-epoch 50 --lambda-warmup-epochs 50 --dataset Houston --shots 20 --experiments baseline name_low_lambda spectral_low_lambda spectral_lidar_low_lambda --low-lambda 0.003 --epochs 200 --iterations 10
```

For a quick validation run:

```powershell
python train_vlm_ablation.py --device cuda --amp --allow-tf32 --center-prototypes --semantic-start-epoch 50 --dataset Houston --shots 20 --experiments baseline spectral_lidar_low_lambda --low-lambda 0.003 --epochs 200 --iterations 1
```

For cross-dataset K-shot evaluation:

```powershell
python train_vlm_ablation.py --device cuda --amp --allow-tf32 --center-prototypes --semantic-start-epoch 50 --dataset Trento --shots 20 --experiments baseline spectral_lidar_low_lambda --low-lambda 0.003 --epochs 200 --iterations 10
python train_vlm_ablation.py --device cuda --amp --allow-tf32 --center-prototypes --semantic-start-epoch 50 --dataset Houston --shots 20 --experiments baseline spectral_lidar_low_lambda --low-lambda 0.003 --epochs 200 --iterations 10
python train_vlm_ablation.py --device cuda --amp --allow-tf32 --center-prototypes --semantic-start-epoch 50 --dataset MUUFL --shots 20 --experiments baseline spectral_lidar_low_lambda --low-lambda 0.003 --epochs 200 --iterations 10
```

Percentage-based Trento evaluation remains available:

```powershell
python train_vlm_ablation.py --device cuda --amp --allow-tf32 --dataset Trento --pcts 0.01 0.05 0.10 --experiments baseline name spectral spectral_lidar --epochs 200 --iterations 10
```

## Useful Options

- `--device cuda` - require CUDA. The script fails if CUDA is unavailable.
- `--amp` - use mixed precision on CUDA.
- `--allow-tf32` - allow faster TF32 operations on Ampere or newer NVIDIA GPUs.
- `--deterministic` - force deterministic cuDNN behavior. This is slower.
- `--center-prototypes` - center text prototypes across classes and normalize again.
- `--semantic-start-epoch 50` - delay semantic loss so the classifier learns first.
- `--lambda-warmup-epochs 50` - gradually increase semantic loss after it starts.
- `--low-lambda 0.003` - lower semantic regularization weight.
- `--shots 10 20 30` - run K-shot settings.
- `--pcts 0.01 0.05 0.10` - run percentage-based few-shot settings.

## Analysis

Aggregate a run:

```powershell
python analyze_runs.py --run-dir runs\Houston_YYYYMMDD_HHMMSS --dataset Houston
```

Generate t-SNE for a K-shot split:

```powershell
python analyze_runs.py --run-dir runs\Houston_YYYYMMDD_HHMMSS --dataset Houston --make-tsne --tsne-label 20-shot --tsne-iter 0
```

Generated analysis files:

- `analysis/aggregate_summary.csv`
- `analysis/paired_deltas.csv`
- `analysis/paired_delta_summary.csv`
- `analysis/oa_trends.png`
- optional `analysis/tsne_cls_*.png`

The t-SNE uses Cross-HL CLS features, not the semantic projection head.

## Notebook

Open:

```text
notebooks/CrossHL_VLM_Showcase.ipynb
```

Recommended notebook settings:

```python
LOAD_CLIP = True
CENTER_PROTOTYPES = True
DATASETS_TO_RUN = ["Trento", "Houston", "MUUFL"]
SHOTS = [20]
SEMANTIC_START_EPOCH = 50
LAMBDA_WARMUP_EPOCHS = 50
RUN_ABLATION = True
```

Use `ITERATIONS = 3` for a short cross-dataset check and `ITERATIONS = 10` for final tables.

## Reproducibility Notes

By default, the runner favors CUDA speed. Add `--deterministic` when exact deterministic cuDNN behavior is required. The random seeds are still controlled for split generation and model initialization.

The semantic branch is a regularizer only. Report `baseline` and semantic variants side by side, and use paired deltas from `analyze_runs.py` when discussing gains or negative transfer.
