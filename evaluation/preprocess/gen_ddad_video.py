import argparse
import os
import sys

import imageio
import numpy as np
from tqdm import tqdm

try:
    from .common import center_crop, depth_to_point_map, normalize_rgb_frame, resolve_device, write_hdf5
except ImportError:
    from common import center_crop, depth_to_point_map, normalize_rgb_frame, resolve_device, write_hdf5


def parse_args():
    parser = argparse.ArgumentParser(description="Generate DDAD benchmark videos/hdf5 for evaluation")
    parser.add_argument("--data_dir", type=str, default="workspace/datasets/DDAD/ddad_train_val")
    parser.add_argument("--output_dir", type=str, default="workspace/benchmark_datasets/DDAD_video")
    parser.add_argument("--dgp_root", type=str, default="evaluation/preprocess/dgp")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--camera_name", type=str, default="CAMERA_01")
    parser.add_argument("--crop_h", type=int, default=1152)
    parser.add_argument("--crop_w", type=int, default=1920)
    parser.add_argument("--depth_eps", type=float, default=1e-5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video_quality", type=int, default=9)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # DDAD loader lives in dgp package; allow path override for portability.
    if args.dgp_root not in sys.path:
        sys.path.append(args.dgp_root)
    from dgp.datasets import SynchronizedSceneDataset

    dataset = SynchronizedSceneDataset(
        os.path.join(args.data_dir, "ddad.json"),
        datum_names=(args.camera_name, "lidar"),
        generate_depth_from_datum="lidar",
        split=args.split,
    )

    seq_list = {}
    # Group sample indices by scene id so each output corresponds to one scene video.
    for scene_idx, sample_idx_in_scene, _ in dataset.dataset_item_index:
        key = str(scene_idx)
        seq_list.setdefault(key, []).append(sample_idx_in_scene)

    meta_infos = []
    for idx, seq_name in enumerate(sorted(seq_list.keys())):
        seq_len = len(seq_list[seq_name])
        seq_out_dir = os.path.join(args.output_dir, args.split, seq_name)
        os.makedirs(seq_out_dir, exist_ok=True)

        st_idx = 0
        ed_idx = seq_len
        video_save_path = os.path.join(seq_out_dir, f"{st_idx:05d}_{ed_idx:05d}_rgb.mp4")
        data_save_path = os.path.join(seq_out_dir, f"{st_idx:05d}_{ed_idx:05d}_data.hdf5")

        meta_infos.append(
            {
                "video": os.path.relpath(video_save_path, args.output_dir),
                "data": os.path.relpath(data_save_path, args.output_dir),
            }
        )

        frames = []
        disps = []
        point_maps = []
        valid_masks = []
        camera_poses = []

        for i in tqdm(range(seq_len), desc=f"DDAD {seq_name} ({idx + 1}/{len(seq_list)})"):
            scene_idx = int(seq_name)
            sample_idx_in_scene = seq_list[seq_name][i]
            # dgp uses lowercase datum names in get_datum_data.
            sample = dataset.get_datum_data(scene_idx, sample_idx_in_scene, args.camera_name.lower())

            img = normalize_rgb_frame(np.array(sample["rgb"]))
            depth = sample["depth"]
            invalid_mask = depth < args.depth_eps

            depth[invalid_mask] = args.depth_eps
            disp = 1.0 / depth
            disp[invalid_mask] = 0.0
            valid_mask = np.logical_not(invalid_mask)

            fx, fy = sample["intrinsics"][0, 0], sample["intrinsics"][1, 1]
            cx, cy = sample["intrinsics"][0, 2], sample["intrinsics"][1, 2]

            point_map = depth_to_point_map(depth, fx, fy, cx, cy, device=device)
            camera_pose = sample["pose"].matrix

            frames.append(img)
            disps.append(disp)
            valid_masks.append(valid_mask)
            point_maps.append(point_map)
            camera_poses.append(camera_pose)

        frames = np.stack(frames)
        disps = np.stack(disps)
        valid_masks = np.stack(valid_masks)
        point_maps = np.stack(point_maps)
        camera_poses = np.stack(camera_poses)

        # Keep benchmark output resolution consistent with prior protocol.
        frames = center_crop(frames, (args.crop_h, args.crop_w))
        disps = center_crop(disps[..., None], (args.crop_h, args.crop_w))[..., 0]
        valid_masks = center_crop(valid_masks[..., None], (args.crop_h, args.crop_w))[..., 0]
        point_maps = center_crop(point_maps, (args.crop_h, args.crop_w))

        imageio.mimsave(video_save_path, frames, fps=args.fps, quality=args.video_quality)
        write_hdf5(data_save_path, disps, valid_masks, point_maps, camera_poses)

    # Export index used by evaluation/eval.py.
    meta_path = os.path.join(args.output_dir, "filename_list.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        for meta in meta_infos:
            print(meta["video"], meta["data"], file=f)

    print(f"Saved metadata list to {meta_path}")


if __name__ == "__main__":
    main()