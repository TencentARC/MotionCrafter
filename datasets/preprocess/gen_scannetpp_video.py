"""Preprocess ScanNet++ RGB-D sequences into MotionCrafter format."""

import numpy as np
import h5py
import os
from PIL import Image
import numpy as np
import torch
import glob
import h5py
from tqdm import tqdm
import imageio
import cv2
import json
from preprocess_common import depth2point_map, get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/scannetpp/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/scannetpp_video/")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def load_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    invalid_mask = depth > 65500
    depth = depth / 1000 # mm -> m
    return depth, invalid_mask


if __name__ == '__main__':
    
    meta_infos = []

    seq_names = sorted(os.listdir(DATA_DIR))

    for idx, seq_name in enumerate(seq_names):
        if seq_name.endswith('.txt'):
            continue
        rgb_paths = sorted(glob.glob(os.path.join(DATA_DIR, seq_name, 'rgb', '*.jpg')))
        rgb_mask_paths = [p.replace('/rgb/', '/rgb_masks/').replace('.jpg', '.png') for p in rgb_paths]
        depth_paths = [p.replace('/rgb/', '/depth/').replace('.jpg', '.png') for p in rgb_paths]
        camera_path = os.path.join(DATA_DIR, seq_name, 'colmap', 'pose_intrinsic_imu.json')
        try:
            with open(camera_path, "r") as f:
                imu_data = json.load(f)
        except Exception as e:
            print(f"Failed to load camera data for sequence {seq_name}: {e}")
            continue

        seq_len = len(rgb_paths)
        progress_bar = tqdm(
            range(seq_len),
        )
        progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

        record = True
        stride = 10
        try:
            for st_idx in range(0, seq_len, CLIP_LENGTH * stride):
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
                camera_poses = []
                valid_masks = []

                for idx in range(st_idx, ed_idx):
                    rgb_path = rgb_paths[idx]
                    depth_path = depth_paths[idx]
                    rgb_mask_path = rgb_mask_paths[idx]

                    img = Image.open(rgb_path)
                    img = np.array(img).astype(np.uint8)
                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]
            
                    intrinsic = imu_data[f'frame_{idx:06d}']['intrinsic']
                    camera_pose = imu_data[f'frame_{idx:06d}']['pose']

                    rgb_mask = cv2.imread(rgb_mask_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) == 255
                    
                    orig_h, orig_w = img.shape[0], img.shape[1]
                    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
                    rgb_mask = cv2.resize(rgb_mask.astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST).astype(bool)

                    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 1000.0
                    depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
                    valid_mask = (depth > DEPTH_EPS) & (np.isinf(depth) == False) & rgb_mask

                    fx, fy = intrinsic[0][0], intrinsic[1][1]
                    cx, cy = intrinsic[0][2], intrinsic[1][2]

                    # downsample intrinsic
                    fx = fx * 640 / orig_w
                    fy = fy * 480 / orig_h
                    cx = cx * 640 / orig_w
                    cy = cy * 480 / orig_h
                    
                    point_map = depth2point_map(
                        torch.from_numpy(depth).float(),
                        fx, fy, cx, cy
                    ) # h,w,3

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

                imageio.mimsave(video_save_path, frames, fps=24, quality=9)
                with h5py.File(data_save_path, 'w') as h5f:
                    h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                    h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
                    h5f.create_dataset('camera_pose', data=camera_poses.astype(np.float16), chunks=(1, )+camera_poses.shape[1:])
                    

        except Exception as e:
            print(f"Failed to process sequence {seq_name}: {e}")
            continue

    write_meta_infos(OUTPUT_DIR, meta_infos)