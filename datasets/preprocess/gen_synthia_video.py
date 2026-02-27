"""Preprocess SYNTHIA video-sequence data into MotionCrafter clips."""

import numpy as np
import h5py
import cv2
import os
from PIL import Image
import numpy as np
import torch
import glob
import h5py
from tqdm import tqdm
import imageio
from preprocess_common import depth2point_map, get_env_int, get_env_list, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/Synthia/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/Synthia_video/")
SPLITS = get_env_list("MOTIONCRAFTER_SPLITS", "video-sequence")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def load_depth(file):
    depth = cv2.imread(str(file), flags=cv2.IMREAD_ANYDEPTH).astype(np.float32)
    invalid_mask = depth >= 64000
    depth = depth / 100.0  # convert to meters
    return depth, invalid_mask

if __name__ == '__main__':
    
    meta_infos = []

    for split in SPLITS:
        if split == 'video-sequence':
            seq_names = sorted(filter(lambda x:x.startswith('SYNTHIA-SEQS') and (not x.endswith('.rar')), os.listdir(os.path.join(DATA_DIR, 'Synthia-Video-Sequence'))))
            for idx, seq_name in enumerate(seq_names):
                cam_names = os.listdir(os.path.join(DATA_DIR, 'Synthia-Video-Sequence', seq_name, 'RGB', 'Stereo_Left'))
                for cam_name in cam_names:
                    rgb_paths = glob.glob(os.path.join(DATA_DIR, 'Synthia-Video-Sequence', seq_name, 'RGB', 'Stereo_Left', cam_name, '*.png'))
                    rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
                    rgb_paths = sorted(rgb_paths)
                    depth_paths = [p.replace('RGB', 'Depth') for p in rgb_paths]
                    meta_paths = ['/'.join(p.split('/')[:-4] + ['CameraParams', 'intrinsics.txt']) for p in rgb_paths]
                    poses_paths = [p.replace('RGB', 'CameraParams').replace('.png', '.txt') for p in rgb_paths]
                    
                    os.makedirs(os.path.join(OUTPUT_DIR, split, seq_name, cam_name), exist_ok=True)
                    
                    seq_len = len(rgb_paths)
                    progress_bar = tqdm(
                        range(seq_len),
                    )
                    progress_bar.set_description(f"Exec {seq_name}|{cam_name} ({idx}/{len(seq_names)})")

                    for st_idx in range(0, seq_len, CLIP_LENGTH):
                        ed_idx = st_idx + CLIP_LENGTH
                        ed_idx = min(ed_idx, seq_len)
                        video_save_path = os.path.join(OUTPUT_DIR, split, seq_name, cam_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
                        data_save_path = os.path.join(OUTPUT_DIR, split, seq_name, cam_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

                        meta_infos.append(dict(
                            video=os.path.relpath(video_save_path, OUTPUT_DIR),
                            data=os.path.relpath(data_save_path, OUTPUT_DIR),
                            num_frames=ed_idx-st_idx
                        ))

                        frames = []
                        # disps = []
                        point_maps = []
                        camera_poses = []
                        valid_masks = []

                        for rgb_path, depth_path, meta_path, pose_path in zip(rgb_paths[st_idx:ed_idx], depth_paths[st_idx:ed_idx], meta_paths[st_idx:ed_idx], poses_paths[st_idx:ed_idx]):

                            if not(os.path.exists(os.path.join(DATA_DIR, rgb_path)) and os.path.exists(os.path.join(DATA_DIR, depth_path)) and os.path.exists(os.path.join(DATA_DIR, meta_path))):
                                continue
                            try: 
                                img = Image.open(os.path.join(DATA_DIR, rgb_path))
                                img = np.array(img).astype(np.uint8)

                                # grayscale images
                                if len(img.shape) == 2:
                                    img = np.tile(img[..., None], (1, 1, 3))
                                else:
                                    img = img[..., :3]

                                depth, invalid_mask = load_depth(os.path.join(DATA_DIR, depth_path))
                                invalid_mask = (depth < DEPTH_EPS) | np.isnan(depth) | np.isinf(depth) | invalid_mask
                                
                                depth[invalid_mask] = DEPTH_EPS
                                # disp = 1.0 / depth
                                # disp[invalid_mask] = 0.

                                valid_mask = np.logical_not(invalid_mask)
                                
                                with open(os.path.join(DATA_DIR, meta_path), 'r') as f:
                                    lines = f.readlines()
                                    filtered_lines = []
                                    for line in lines:
                                        if line.strip() != '':
                                            filtered_lines.append(line)
                                focal_length = float(filtered_lines[0])
                                cx = float(filtered_lines[1])
                                cy = float(filtered_lines[2])

                                with open(os.path.join(DATA_DIR, pose_path), 'r') as f:
                                    line = f.readline()
                                camera_pose = np.array([float(x) for x in line.split()]).reshape(4, 4).T
                                T_flip_z = np.diag([1, 1, -1, 1])  # flip Z-axis
                                camera_pose = camera_pose @ T_flip_z

                                if img.shape[0] % 16 != 0  or img.shape[1] % 16 != 0:
                                    H = img.shape[0] // 16 * 16
                                    W = img.shape[1] // 16 * 16
                                    x0, y0 = (img.shape[1] - W) // 2, (img.shape[0] - H) // 2
                                    img = img[y0:y0+H, x0:x0+W, :]
                                    depth = depth[y0:y0+H, x0:x0+W]
                                    # disp = disp[y0:y0+H, x0:x0+W]
                                    valid_mask = valid_mask[y0:y0+H, x0:x0+W]
                                    cx -= x0
                                    cy -= y0

                                focal_length = np.array([focal_length, focal_length])
                                principal_point = np.array([cx, cy])
                                point_map = depth2point_map(torch.tensor(depth).float().cuda(), focal_length[0], focal_length[1], 
                                                            principal_point[0], principal_point[1])
                                
                                frames.append(img)
                                # disps.append(disp)
                                valid_masks.append(valid_mask)
                                point_maps.append(point_map.cpu().numpy())
                                camera_poses.append(camera_pose)
                            except Exception as e:
                                print(e)

                            progress_bar.update(1)
                        
                        frames = np.stack(frames)
                        # disps = np.stack(disps)
                        valid_masks = np.stack(valid_masks)
                        point_maps = np.stack(point_maps)
                        camera_poses = np.stack(camera_poses)

                        imageio.mimsave(video_save_path, frames, fps=24, quality=9)
                        with h5py.File(data_save_path, 'w') as h5f:
                            # h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
                            h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                            h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
                            h5f.create_dataset('camera_pose', data=camera_poses.astype(np.float16), chunks=(1, )+camera_poses.shape[1:])

    write_meta_infos(OUTPUT_DIR, meta_infos)