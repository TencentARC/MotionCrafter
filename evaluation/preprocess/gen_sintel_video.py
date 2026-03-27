import argparse
import glob
import os

import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from .common import center_crop, depth_to_point_map, normalize_rgb_frame, resolve_device, write_hdf5
except ImportError:
    from common import center_crop, depth_to_point_map, normalize_rgb_frame, resolve_device, write_hdf5


TAG_FLOAT = 202021.25


def cam_read(filename):
    # Sintel .cam format starts with a magic float tag.
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, f"cam_read wrong tag: {check}"
        intr = np.fromfile(f, dtype="float64", count=9).reshape((3, 3))
        extr = np.fromfile(f, dtype="float64", count=12).reshape((3, 4))
    return intr, extr


def depth_read(filename):
    # Sintel .dpt format: magic tag + width + height + raw float32 depth.
    with open(filename, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        assert check == TAG_FLOAT, f"depth_read wrong tag: {check}"
        width = np.fromfile(f, dtype=np.int32, count=1)[0]
        height = np.fromfile(f, dtype=np.int32, count=1)[0]
        size = width * height
        assert width > 0 and height > 0 and 1 < size < 100000000
        depth = np.fromfile(f, dtype=np.float32, count=-1).reshape((height, width))
    return depth


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Sintel benchmark videos/hdf5 for evaluation")
    parser.add_argument("--data_dir", type=str, default="workspace/datasets/SintelComplete")
    parser.add_argument("--output_dir", type=str, default="workspace/benchmark_datasets/Sintel_video")
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--crop_h", type=int, default=436)
    parser.add_argument("--crop_w", type=int, default=872)
    parser.add_argument("--depth_eps", type=float, default=1e-5)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--video_quality", type=int, default=9)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)

    meta_infos = []
    clean_root = os.path.join(args.data_dir, args.split, "clean")
    seq_names = sorted(os.listdir(clean_root))

    for idx, seq_name in enumerate(seq_names):
        # Build aligned lists for RGB / depth / invalid mask / camera files.
        rgb_paths = glob.glob(os.path.join(clean_root, seq_name, "frame_*.png"))
        rgb_paths = [os.path.relpath(p, args.data_dir) for p in rgb_paths]
        rgb_paths = sorted(rgb_paths, key=lambda p: int(os.path.basename(p).split("_")[-1][:-4]))

        depth_paths = [p.replace("clean/", "depth/").replace(".png", ".dpt") for p in rgb_paths]
        mask_paths = [p.replace("clean/", "mask/") for p in rgb_paths]
        meta_paths = [p.replace("clean/", "camdata_left/").replace(".png", ".cam") for p in rgb_paths]

        seq_out_dir = os.path.join(args.output_dir, seq_name)
        os.makedirs(seq_out_dir, exist_ok=True)

        st_idx = 0
        ed_idx = len(rgb_paths)
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

        iterator = zip(
            rgb_paths[st_idx:ed_idx],
            depth_paths[st_idx:ed_idx],
            mask_paths[st_idx:ed_idx],
            meta_paths[st_idx:ed_idx],
        )
        for rgb_path, depth_path, mask_path, meta_path in tqdm(
            iterator,
            total=ed_idx - st_idx,
            desc=f"Sintel {seq_name} ({idx + 1}/{len(seq_names)})",
        ):
            img = np.array(Image.open(os.path.join(args.data_dir, rgb_path)))
            img = normalize_rgb_frame(img)

            depth = depth_read(os.path.join(args.data_dir, depth_path))
            invalid_mask = np.array(Image.open(os.path.join(args.data_dir, mask_path))).astype(np.uint8) > 127
            invalid_mask = np.logical_or(depth < args.depth_eps, invalid_mask)

            depth[invalid_mask] = args.depth_eps
            disp = 1.0 / depth
            disp[invalid_mask] = 0.0
            valid_mask = np.logical_not(invalid_mask)

            intr, extr = cam_read(os.path.join(args.data_dir, meta_path))
            fx, fy = intr[0, 0], intr[1, 1]
            cx, cy = intr[0, 2], intr[1, 2]

            # Convert extrinsic [R|t] to 4x4 camera-to-world.
            camera_pose = np.eye(4, dtype=np.float64)
            camera_pose[:3, :4] = extr
            camera_pose = np.linalg.inv(camera_pose)

            # Keep even resolution (some video codecs/processing tools assume even sizes).
            if img.shape[0] % 2 != 0 or img.shape[1] % 2 != 0:
                h = img.shape[0] // 2 * 2
                w = img.shape[1] // 2 * 2
                x0 = (img.shape[1] - w) // 2
                y0 = (img.shape[0] - h) // 2
                img = img[y0 : y0 + h, x0 : x0 + w, :]
                depth = depth[y0 : y0 + h, x0 : x0 + w]
                disp = disp[y0 : y0 + h, x0 : x0 + w]
                valid_mask = valid_mask[y0 : y0 + h, x0 : x0 + w]
                cx -= x0
                cy -= y0

            point_map = depth_to_point_map(depth, fx, fy, cx, cy, device=device)

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

        # Match benchmark resolution by center cropping all tensors consistently.
        frames = center_crop(frames, (args.crop_h, args.crop_w))
        disps = center_crop(disps[..., None], (args.crop_h, args.crop_w))[..., 0]
        valid_masks = center_crop(valid_masks[..., None], (args.crop_h, args.crop_w))[..., 0]
        point_maps = center_crop(point_maps, (args.crop_h, args.crop_w))

        imageio.mimsave(
            video_save_path,
            frames,
            fps=args.fps,
            quality=args.video_quality,
            macro_block_size=1,
        )
        write_hdf5(data_save_path, disps, valid_masks, point_maps, camera_poses)

    # This file is the evaluator input index (video_path data_path per line).
    meta_path = os.path.join(args.output_dir, "filename_list.txt")
    with open(meta_path, "w", encoding="utf-8") as f:
        for meta in meta_infos:
            print(meta["video"], meta["data"], file=f)

    print(f"Saved metadata list to {meta_path}")


if __name__ == "__main__":
    main()