"""Build latent-space meta_infos.txt from source preprocess metadata."""

import argparse
import glob
import os
from typing import List, Set


def parse_meta_sample_dirs(meta_file_path: str) -> Set[str]:
    sample_dirs: Set[str] = set()
    with open(meta_file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            _, data_path = parts[:2]
            sample_dirs.add(os.path.dirname(data_path))
    return sample_dirs


def collect_npz_samples(latent_dir: str, sample_dirs: List[str]) -> List[str]:
    samples: List[str] = []
    for sample_dir in sample_dirs:
        pattern = os.path.join(latent_dir, sample_dir, "*.npz")
        samples.extend(glob.glob(pattern))
    return sorted(samples)


def build_meta_infos(data_dir: str, latent_dir: str) -> int:
    meta_file_path = os.path.join(data_dir, "meta_infos.txt")
    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(f"meta file not found: {meta_file_path}")
    if not os.path.exists(latent_dir):
        raise FileNotFoundError(f"latent dir not found: {latent_dir}")

    sample_dirs = sorted(parse_meta_sample_dirs(meta_file_path))
    samples = collect_npz_samples(latent_dir, sample_dirs)

    output_meta_path = os.path.join(latent_dir, "meta_infos.txt")
    with open(output_meta_path, "w", encoding="utf-8") as file:
        for sample in samples:
            print("NA", os.path.relpath(sample, latent_dir), "0", file=file)

    print(f"{latent_dir} found {len(samples)} samples")
    return len(samples)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate latent meta_infos.txt from source dataset meta_infos.txt"
    )
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="One or more source dataset directories containing meta_infos.txt",
    )
    parser.add_argument(
        "--latent-dirs",
        nargs="+",
        required=True,
        help="One or more latent dataset directories aligned with --data-dirs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.data_dirs) != len(args.latent_dirs):
        raise ValueError("--data-dirs and --latent-dirs must have the same length")

    for data_dir, latent_dir in zip(args.data_dirs, args.latent_dirs):
        build_meta_infos(data_dir=data_dir, latent_dir=latent_dir)


if __name__ == "__main__":
    main()
