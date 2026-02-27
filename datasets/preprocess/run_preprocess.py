"""Unified command-line launcher for dataset-specific preprocess scripts."""

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_REGISTRY = {
    "blinkvision": "gen_blinkvision_video.py",
    "dynamic_replica": "gen_dynamic_replica_video.py",
    "gta_sfm": "gen_gta_sfm_video.py",
    "kubric": "gen_kubric_video.py",
    "matrix_city": "gen_matrix_city_video.py",
    "mvs_synth": "gen_mvs_synth_video.py",
    "omniworld": "gen_omniworld_video.py",
    "point_odyssey": "gen_point_odessey_video.py",
    "scannetpp": "gen_scannetpp_video.py",
    "spring": "gen_spring_video.py",
    "synthia": "gen_synthia_video.py",
    "tartan_air": "gen_tartan_air_video.py",
    "virtual_kitti2": "gen_virtual_kitti2_video.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset preprocess script by dataset name.")
    parser.add_argument("--dataset", choices=sorted(SCRIPT_REGISTRY.keys()), required=True)
    parser.add_argument("--data-dir", type=str, required=True, help="Raw dataset root.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for preprocessed data.")
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, help="Single split for scripts using MOTIONCRAFTER_SPLIT.")
    parser.add_argument("--splits", type=str, default=None, help="Comma-separated splits for scripts using MOTIONCRAFTER_SPLITS.")
    parser.add_argument(
        "--process-scene-flow",
        action="store_true",
        help="Enable scene flow generation for scripts that support it (e.g. Point Odyssey).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_name = SCRIPT_REGISTRY[args.dataset]

    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"script not found: {script_path}")

    env = os.environ.copy()
    env["MOTIONCRAFTER_DATA_DIR"] = args.data_dir
    env["MOTIONCRAFTER_OUTPUT_DIR"] = args.output_dir

    if args.clip_length is not None:
        env["MOTIONCRAFTER_CLIP_LENGTH"] = str(args.clip_length)
    if args.split is not None:
        env["MOTIONCRAFTER_SPLIT"] = args.split
    if args.splits is not None:
        env["MOTIONCRAFTER_SPLITS"] = args.splits
    if args.process_scene_flow:
        env["MOTIONCRAFTER_PROCESS_SCENE_FLOW"] = "true"

    command = [sys.executable, str(script_path)]
    print("Run:", " ".join(command))
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
