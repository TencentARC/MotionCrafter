"""Preprocess BlinkVision sequences into MotionCrafter training format."""

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
import json
from scipy.spatial.transform import Rotation as R
from preprocess_common import depth2point_map, get_env_int, get_env_str, write_meta_infos


DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/blinkvision/indoor_train")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/blinkvision_video/")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)


blender2opencv = np.float32([[1, 0, 0, 0],
                             [0, -1, 0, 0],
                             [0, 0, -1, 0],
                             [0, 0, 0, 1]])

def convert_to_w2c(frame): # to get world-to-camera matrix
    translation = frame[0] 
    rotation = frame[1] 
    
    RR = R.from_euler('xyz', rotation, degrees=False).as_matrix()
    w2c = np.eye(4)
    w2c[:3, :3] = RR.T
    w2c[:3, 3] = -RR.T @ translation

    w2c = blender2opencv @ w2c
    return w2c


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
        if seq_name.endswith('.txt'):
            continue
        rgb_paths = sorted(glob.glob(os.path.join(DATA_DIR, seq_name, 'clean_uint8', '*.png')))
        depth_paths = [p.replace('/clean_uint8/', '/depth/').replace('.png', '.npz') for p in rgb_paths]
        meta_path = os.path.join(DATA_DIR, seq_name, 'metadata.json')
        pose_path = os.path.join(DATA_DIR, seq_name, 'poses.npz')
        poses = np.load(pose_path)['camera_poses']
        print(poses.shape)

        with open(meta_path, 'r') as f:
            meta = json.load(f)
            intrinsics = np.array(meta['K_matrix'])

        seq_len = len(rgb_paths)
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
                rgb_path = rgb_paths[idx]
                depth_path = depth_paths[idx]

                img = Image.open(rgb_path)
                img = np.array(img).astype(np.uint8)
                # grayscale images
                if len(img.shape) == 2:
                    img = np.tile(img[..., None], (1, 1, 3))
                else:
                    img = img[..., :3]

                depth = np.load(depth_path)['depth']
                valid_mask = (depth < 30) & (depth > 1e-3)

                fx, fy = intrinsics[0,0], intrinsics[1,1]
                cx, cy = intrinsics[0,2], intrinsics[1,2]
                w2c = convert_to_w2c(poses[idx])  
                w2c[:3, 3] = w2c[:3, 3] / 100
                camera_pose = np.linalg.inv(np.array(w2c)) # camera to world

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

    write_meta_infos(OUTPUT_DIR, meta_infos)