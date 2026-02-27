# Preprocess Scripts (Open-source Usage)

This folder contains dataset-specific scripts that convert raw datasets to MotionCrafter training format:
- video clips: `*_rgb.mp4`
- annotations: `*_data.hdf5`
- index file: `meta_infos.txt`

## 0) Unified launcher (recommended)

Use one command to run a dataset-specific preprocess script:

```bash
python run_preprocess.py \
  --dataset spring \
  --data-dir /path/to/raw/Spring \
  --output-dir /path/to/unnormed/Spring_video \
  --split train \
  --clip-length 150
```

For scripts using multi-splits:

```bash
python run_preprocess.py \
  --dataset gta_sfm \
  --data-dir /path/to/raw/GTA-SfM \
  --output-dir /path/to/unnormed/GTA-SfM_video \
  --splits train,test
```

## 1) Common configuration via environment variables

All `gen_*.py` scripts now support the same environment variables:

- `MOTIONCRAFTER_DATA_DIR`: raw dataset root directory
- `MOTIONCRAFTER_OUTPUT_DIR`: output directory for converted videos/hdf5
- `MOTIONCRAFTER_CLIP_LENGTH`: frames per clip

Optional variables used by some scripts:

- `MOTIONCRAFTER_SPLITS`: comma-separated split list (e.g. `train,test`)
- `MOTIONCRAFTER_SPLIT`: single split name
- `MOTIONCRAFTER_PROCESS_SCENE_FLOW`: `true/false` (Point Odyssey)

### Example

```bash
cd datasets/preprocess
MOTIONCRAFTER_DATA_DIR=/path/to/raw/Spring \
MOTIONCRAFTER_OUTPUT_DIR=/path/to/unnormed/Spring_video \
MOTIONCRAFTER_SPLIT=train \
MOTIONCRAFTER_CLIP_LENGTH=150 \
python gen_spring_video.py
```

## 2) Normalize generated datasets

`normalize_video_dataset.py` supports CLI arguments:

```bash
python normalize_video_dataset.py \
  --data-dirs /path/to/unnormed/Spring_video /path/to/unnormed/GTA-SfM_video \
  --output-root /path/to/tmp_datasets \
  --resolution 320 640 \
  --num-workers 8 \
  --skip-existing
```

If `--output-root` and `--output-dirs` are not provided, output defaults to:
- replace `unnormed_datasets` with `tmp_datasets` in `data_dir`, or
- `data_dir/normalized` when replacement is not possible.

## 3) Build latent meta file

`preprocess_meta_file.py` generates latent `meta_infos.txt` from source meta files:

```bash
python preprocess_meta_file.py \
  --data-dirs /path/to/data_normed_1/Spring_video /path/to/data_normed_1/GTA-SfM_video \
  --latent-dirs /path/to/latent/Spring /path/to/latent/GTA-SfM
```

## 4) Notes

- Most scripts assume CUDA is available for point map / flow computation.
- Keep clip length consistent with your training setup.
- Some datasets require dataset-specific directory structures exactly as expected by each script.
- Common helper functions for env parsing and `meta_infos.txt` writing are in `preprocess_common.py`.

## 5) Dataset-specific dependencies

- Dynamic Replica: requires `pytorch3d`.
- Virtual KITTI 2: requires `pandas`.
- Scripts reading EXR files (e.g. IRS/Matrix City): require `OpenEXR` runtime support.

If `pip install -r requirements.txt` cannot install `pytorch3d` in your environment,
install it separately by following the official PyTorch3D installation guide that matches
your PyTorch and CUDA versions.
