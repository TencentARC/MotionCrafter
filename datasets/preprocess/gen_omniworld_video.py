"""Preprocess OmniWorld splits into MotionCrafter training artifacts."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
import glob
from tqdm import tqdm
import imageio
import json
import random
from scipy.spatial.transform import Rotation as R
import imageio.v2 as iio
from preprocess_common import depth2point_map, get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/OmniWorld")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/Omniworld_video/")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def load_depth(depthpath):
    """
    Returns
    -------
    depthmap : (H, W) float32
    valid   : (H, W) bool      True for reliable pixels
    """

    depthmap = iio.imread(depthpath).astype(np.float32) / 65535.0
    near_mask = depthmap < 0.0015   # 1. too close
    far_mask = depthmap > (65500.0 / 65535.0) # 2. filter sky
    # far_mask = depthmap > np.percentile(depthmap[~far_mask], 95) # 3. filter far area (optional)
    near, far = 1., 1000.
    depthmap = depthmap / (far - depthmap * (far - near)) / 0.04  # Scale constant, verify for new datasets if needed.

    valid = ~(near_mask | far_mask)
    depthmap[~valid] = -1

    return depthmap, valid

def load_split_info(scene_dir):
    """Return the split json dict."""
    with open(scene_dir + "/split_info.json", "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_camera_poses(scene_dir, split_idx):
    """
    Returns
    -------
    intrinsics : (S, 3, 3) array, pixel-space K matrices
    extrinsics : (S, 4, 4) array, OpenCV world-to-camera matrices
    """
    # ----- read metadata -----------------------------------------------------
    split_info = load_split_info(scene_dir)
    frame_count = len(split_info["split"][split_idx])

    cam_file = scene_dir + "/camera" + f"/split_{split_idx}.json"
    with open(cam_file, "r", encoding="utf-8") as f:
        cam = json.load(f)

    # ----- intrinsics --------------------------------------------------------
    intrinsics = np.repeat(np.eye(3)[None, ...], frame_count, axis=0)
    intrinsics[:, 0, 0] = cam["focals"]          # fx
    intrinsics[:, 1, 1] = cam["focals"]          # fy
    intrinsics[:, 0, 2] = cam["cx"]              # cx
    intrinsics[:, 1, 2] = cam["cy"]              # cy

    # ----- extrinsics --------------------------------------------------------
    extrinsics = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)

    # SciPy expects quaternions as (x, y, z, w) → convert
    quat_wxyz = np.array(cam["quats"])           # (S, 4)  (w,x,y,z)
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)

    rotations = R.from_quat(quat_xyzw).as_matrix()      # (S, 3, 3)
    translations = np.array(cam["trans"])               # (S, 3)

    extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = translations

    return intrinsics.astype(np.float32), extrinsics.astype(np.float32)


def center_crop(tensor, cropped_size):
    size = tensor.shape[1:3]
    height, width = size
    cropped_height, cropped_width = cropped_size
    needs_vert_crop, top = (
        (True, (height - cropped_height) // 2)
        if height > cropped_height
        else (False, 0)
    )
    needs_horz_crop, left = (
        (True, (width - cropped_width) // 2)
        if width > cropped_width
        else (False, 0)
    )

    needs_crop=needs_vert_crop or needs_horz_crop
    if needs_crop:
        return tensor[:, top:top+cropped_height, left:left+cropped_width]

if __name__ == '__main__':
    
    meta_infos = []

    seq_names = sorted(os.listdir(DATA_DIR))

    for idx, seq_name in enumerate(seq_names):
        if seq_name.endswith('.txt') or seq_name.endswith('.md') or seq_name.endswith('.gitattributes'):
            continue

        try:        
            rgb_paths = sorted(glob.glob(os.path.join(DATA_DIR, seq_name, 'color', '*.png')))
            depth_paths = [p.replace('/color/', '/depth/') for p in rgb_paths]
            split_path = os.path.join(DATA_DIR, seq_name)
            split_info = load_split_info(split_path)
            split_idx = random.choice(np.arange(split_info['split_num']))
            selected_frames = split_info["split"][split_idx]
            intrinsics, extrinsics = load_camera_poses(split_path, split_idx)
            if selected_frames[-1] - selected_frames[0] + 1 != len(selected_frames):
                print(f"Skipping sequence {seq_name} as it has non-consecutive frames.")
                continue

            seq_len = len(selected_frames)
            progress_bar = tqdm(
                range(seq_len),
            )
            progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

            record = True

            for st_idx in range(0, seq_len, CLIP_LENGTH):
                ed_idx = st_idx + CLIP_LENGTH
                ed_idx = min(ed_idx, seq_len)
                video_save_path = os.path.join(OUTPUT_DIR, seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
                data_save_path = os.path.join(OUTPUT_DIR, seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

                meta_infos.append(dict(
                    video=os.path.relpath(video_save_path, OUTPUT_DIR),
                    data=os.path.relpath(data_save_path, OUTPUT_DIR),
                    num_frames=ed_idx-st_idx
                ))

                frames = []
                point_maps = []
                world_maps = []
                valid_masks = []
                camera_poses = []

                for idx in range(st_idx, ed_idx):
                    rgb_path = rgb_paths[idx + selected_frames[0]]
                    depth_path = depth_paths[idx + selected_frames[0]]

                    img = Image.open(rgb_path)
                    img = np.array(img).astype(np.uint8)
                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]

                    depth, valid_mask = load_depth(depth_path)

                    intrinsic = intrinsics[idx]
                    fx, fy = intrinsic[0,0], intrinsic[1,1]
                    cx, cy = intrinsic[0,2], intrinsic[1,2]
                    camera_pose = np.linalg.inv(extrinsics[idx])  # camera to world

                    point_map = depth2point_map(torch.from_numpy(depth).float(), fx, fy, cx, cy)  # h,w,3

                    frames.append(img)
                    valid_masks.append(valid_mask)
                    point_maps.append(point_map.cpu().numpy())
                    camera_poses.append(camera_pose)

                    progress_bar.update(1)
                
                if record is False:
                    continue
                
                os.makedirs(os.path.join(OUTPUT_DIR, seq_name), exist_ok=True)

                frames = np.stack(frames)
                valid_masks = np.stack(valid_masks)
                point_maps = np.stack(point_maps)
                camera_poses = np.stack(camera_poses)

                h, w = frames.shape[1:3]
                h_crop = h // 16 * 16
                w_crop = w // 16 * 16
                if h_crop != h or w_crop != w:
                    frames = center_crop(frames, (h_crop, w_crop))
                    valid_masks = center_crop(valid_masks, (h_crop, w_crop))
                    point_maps = center_crop(point_maps, (h_crop, w_crop))

                imageio.mimsave(video_save_path, frames, fps=24, quality=9)
                with h5py.File(data_save_path, 'w') as h5f:
                    h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                    h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
                    h5f.create_dataset('camera_pose', data=camera_poses.astype(np.float16), chunks=(1, )+camera_poses.shape[1:])     

        except Exception as e:
            print(f"Error processing sequence {seq_name}: {e}")
            continue

    write_meta_infos(OUTPUT_DIR, meta_infos)