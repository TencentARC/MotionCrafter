import argparse
import functools
import glob
import os

import imageio
import numpy as np
from PIL import Image
from torchvision.datasets.utils import _read_pfm
from tqdm import tqdm

try:
    from .common import depth_to_point_map, normalize_rgb_frame, resolve_device, write_hdf5
except ImportError:
    from common import depth_to_point_map, normalize_rgb_frame, resolve_device, write_hdf5


_read_pfm_file = functools.partial(_read_pfm, slice_channels=1)


def read_camera_pose(filename):
    # Monkaa camera_data.txt stores per-frame left/right 4x4 poses; we read left camera only.
    poses_l = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Frame"):
            l_line = lines[i + 1]
            if l_line.startswith("L"):
                nums = list(map(float, l_line.split()[1:]))
                assert len(nums) == 16, f"Expect 16 numbers, got {len(nums)}"
                pose = np.array(nums, dtype=np.float64).reshape(4, 4)
                # Convert to camera-to-world for consistency with evaluator.
                pose = np.linalg.inv(pose)
                poses_l.append(pose)
            i += 3
        else:
            i += 1
    return poses_l


def parse_scene_thresholds(scene_thres_str):
    # CLI string format: "scene_a:100,scene_b:200".
    scene_thres = {}
    if not scene_thres_str:
        return scene_thres
    for item in scene_thres_str.split(","):
        key, value = item.split(":")
        scene_thres[key] = float(value)
    return scene_thres


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Monkaa benchmark videos/hdf5 for evaluation")
    parser.add_argument("--data_dir", type=str, default="workspace/datasets/SceneFlowDataset/Monkaa")
    parser.add_argument("--output_dir", type=str, default="workspace/benchmark_datasets/Monkaa_video")
    parser.add_argument("--max_seq_len", type=int, default=110)
    parser.add_argument("--depth_eps", type=float, default=1e-5)
    parser.add_argument("--disp_eps", type=float, default=1.0)
    parser.add_argument("--focal_length", type=float, default=1050.0)
    parser.add_argument("--principal_point_x", type=float, default=479.5)
    parser.add_argument("--principal_point_y", type=float, default=269.5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video_quality", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--scene_thresholds",
        type=str,
        default="a_rain_of_stones_x2:300,funnyworld_x2:100,eating_x2:70,family_x2:70,flower_storm_x2:300,lonetree_x2:100,top_view_x2:100,treeflight_x2:300",
        help="Comma-separated scene:threshold list",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    scene_thres = parse_scene_thresholds(args.scene_thresholds)

    meta_infos = []
    scenes = sorted(os.listdir(os.path.join(args.data_dir, "frames_cleanpass")))

    for scene in tqdm(scenes, desc="Monkaa scenes"):
        # Optional allowlist: skip scenes not configured in threshold map.
        if scene_thres and scene not in scene_thres:
            continue

        rgb_paths = glob.glob(os.path.join(args.data_dir, "frames_cleanpass", scene, "left", "*.png"))
        rgb_paths = [os.path.relpath(p, args.data_dir) for p in rgb_paths]
        rgb_paths = sorted(rgb_paths, key=lambda x: int(os.path.basename(x).split(".")[0]))
        rgb_paths = rgb_paths[: args.max_seq_len]
        depth_paths = [p.replace("frames_cleanpass", "disparity").replace(".png", ".pfm") for p in rgb_paths]

        st_idx = 0
        ed_idx = len(rgb_paths)
        os.makedirs(os.path.join(args.output_dir, scene), exist_ok=True)
        video_save_path = os.path.join(args.output_dir, scene, f"{st_idx:05d}_{ed_idx:05d}_rgb.mp4")
        data_save_path = os.path.join(args.output_dir, scene, f"{st_idx:05d}_{ed_idx:05d}_data.hdf5")

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

        camera_pose = read_camera_pose(os.path.join(args.data_dir, "camera_data", scene, "camera_data.txt"))
        camera_pose = camera_pose[: args.max_seq_len]

        for rgb_path, depth_path in zip(rgb_paths, depth_paths):
            img = np.array(Image.open(os.path.join(args.data_dir, rgb_path)))
            img = normalize_rgb_frame(img)

            # PFM stores disparity; convert to depth via z = f / disp.
            disp = _read_pfm_file(os.path.join(args.data_dir, depth_path))[0]
            invalid_mask = disp < args.disp_eps
            disp = np.clip(disp, 1e-3, None)

            fx = fy = args.focal_length
            cx, cy = args.principal_point_x, args.principal_point_y
            depth = args.focal_length / disp
            if scene in scene_thres:
                # Remove far points per-scene to suppress noisy long-range depth.
                invalid_mask = np.logical_or(invalid_mask, depth > scene_thres[scene])

            depth[invalid_mask] = args.depth_eps
            disp[invalid_mask] = 0.0
            disp = disp / img.shape[1]
            valid_mask = np.logical_not(invalid_mask)

            point_map = depth_to_point_map(depth, fx, fy, cx, cy, device=device)

            frames.append(img)
            disps.append(disp)
            valid_masks.append(valid_mask)
            point_maps.append(point_map)

        frames = np.stack(frames)
        disps = np.stack(disps)
        valid_masks = np.stack(valid_masks)
        point_maps = np.stack(point_maps)
        camera_pose = np.stack(camera_pose)

        imageio.mimsave(video_save_path, frames, fps=args.fps, quality=args.video_quality, macro_block_size=1)
        write_hdf5(data_save_path, disps, valid_masks, point_maps, camera_pose)

    # Export index used by evaluation/eval.py.
    meta_path = os.path.join(args.output_dir, "filename_list.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        for meta in meta_infos:
            print(meta["video"], meta["data"], file=f)

    print(f"Saved metadata list to {meta_path}")


if __name__ == "__main__":
    main()