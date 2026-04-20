# Evaluation Pipeline

This directory contains scripts for preparing benchmark datasets and evaluating predicted 4D geometry / motion outputs.

## Directory Structure

- `eval.py`: Main evaluation entrypoint (single dataset directory).
- `metrics.py`: Metric implementations used by `eval.py`.
- `preprocess/`: Dataset-specific preprocessing scripts.

## Data Convention

For each sample, the evaluator expects:

1. Prediction `.npz` under prediction root:
   - `point_map`: `[T, H, W, 3]`
   - optional `scene_flow`: `[T, H, W, 3]`

2. Ground-truth `.hdf5` under GT root:
   - `point_map`: `[T, H, W, 3]`
   - `valid_mask`: `[T, H, W]`
   - `camera_pose`: `[T, 4, 4]` (camera-to-world)
   - optional `scene_flow`: `[T, H, W, 3]`
   - optional `deform_mask`: `[T, H, W]`

3. Metadata file under GT root:
   - `filename_list.txt` (default)
   - `meta_infos.txt` when using `--use_normed_data`

## Benchmark Splits (Reference Only)

The uploaded file `evaluation/benchmark_datasets_splits.zip` is provided only for reference and comparison.

- It contains split list files under `benchmark_datasets_lists/<dataset_name>/`.
- It is **not** directly consumed by `evaluation/eval.py`.
- The evaluator still reads metadata from `--gt_data_dir`:
   - `filename_list.txt` by default
   - `meta_infos.txt` when `--use_normed_data` is enabled

## Single Dataset Evaluation

```bash
python evaluation/eval.py \
  --gt_data_dir workspace/benchmark_datasets/Virtual_KITTI_2_video \
  --pred_data_dir workspace/benchmark_outputs/MotionCrafter/Virtual_KITTI_2_video \
  --use_normed_data \
  --is_pred_world_map
```

### Useful Options

- `--device {auto,cuda,cpu}`: Choose runtime device.
- `--strict_missing`: Fail immediately on missing files.
- `--save_aligned_world`: Save aligned world-space predictions as `*_aligned_world.npz`.
- `--static_pose_for_flow`: Use same pose for flow transform (useful for some datasets).
- `--max_frames_no_flow` / `--max_frames_with_flow`: Frame caps for evaluation.


## Dataset Preprocess

The scripts under `evaluation/preprocess` now support CLI arguments so paths and devices are configurable.

Sintel:

```bash
python evaluation/preprocess/gen_sintel_video.py \
   --data_dir workspace/datasets/SintelComplete \
   --output_dir workspace/benchmark_datasets/Sintel_video \
   --device auto
```

Monkaa:

```bash
python evaluation/preprocess/gen_monkaa_video.py \
   --data_dir workspace/datasets/SceneFlowDataset/Monkaa \
   --output_dir workspace/benchmark_datasets/Monkaa_video \
   --device auto
```

DDAD:

```bash
python evaluation/preprocess/gen_ddad_video.py \
   --data_dir workspace/datasets/DDAD/ddad_train_val \
   --output_dir workspace/benchmark_datasets/DDAD_video \
   --dgp_root evaluation/preprocess/dgp \
   --device auto
```

All preprocess scripts will generate:

- dataset videos and hdf5 files under the output directory
- `filename_list.txt` for downstream evaluator input

## Output JSON

The evaluator writes `metrics.json` to `--pred_data_dir` (or `--save_file_name`).

It includes:

- global metric means
- per-sample metric list
- `_meta` summary (`num_samples_total`, `num_samples_evaluated`, `num_samples_skipped`, `device`)
- `_skipped` entries when files are missing and `--strict_missing` is not set
