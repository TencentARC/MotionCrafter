"""Preprocess TartanAir trajectories into MotionCrafter clip shards."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
import glob
from tqdm import tqdm
import imageio
from preprocess_common import depth2point_map, get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/TartanAir/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/TartanAir_video/")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert a quaternion to a rotation matrix."""
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)],
        [2 * (qx*qy + qw*qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qw*qx)],
        [2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2 * (qx**2 + qy**2)]
    ])
    return R


def ned_to_standard(tx, ty, tz, qx, qy, qz, qw):
    """
    Convert NED frame pose (tx, ty, tz, qx, qy, qz, qw) to standard coordinate frame.
    """
    # Step 1: Flip the Z axis for position
    tz = -tz
    
    # Step 2: Adjust the quaternion (flip qz)
    qz = -qz
    
    # Step 3: Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    
    # Step 4: Construct the 4x4 transformation matrix
    T = np.array([tx, ty, tz])
    
    # 4x4 Transformation matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = R
    pose_matrix[:3, 3] = T
    
    return pose_matrix


if __name__ == '__main__':
    
    meta_infos = []
    seq_names = sorted(filter(lambda x: '.' not in x, os.listdir(DATA_DIR)))

    for idx, seq_name in enumerate(seq_names):
        for level in ['Easy', 'Hard']:
            for video_id in sorted(os.listdir(os.path.join(DATA_DIR, seq_name, level))):
                rgb_paths = glob.glob(os.path.join(DATA_DIR, seq_name, level, video_id, 'image_left', '*_left.png'))
                rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
                rgb_paths = sorted(rgb_paths)
                depth_paths = [p.replace('image_left', 'depth_left').replace('_left.png', '_left_depth.npy') for p in rgb_paths]
                pose_path = os.path.join(DATA_DIR, seq_name, level, video_id, 'pose_left.txt')

                with open(pose_path, 'r') as f:
                    lines = f.readlines()
                
                os.makedirs(os.path.join(OUTPUT_DIR, seq_name, level, video_id), exist_ok=True)
                
                seq_len = len(rgb_paths)
                progress_bar = tqdm(
                    range(seq_len),
                )
                progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

                for st_idx in range(0, seq_len, CLIP_LENGTH):
                    ed_idx = st_idx + CLIP_LENGTH
                    ed_idx = min(ed_idx, seq_len)
                    video_save_path = os.path.join(OUTPUT_DIR, seq_name, level, video_id, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
                    data_save_path = os.path.join(OUTPUT_DIR, seq_name, level, video_id, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

                    meta_infos.append(dict(
                        video=os.path.relpath(video_save_path, OUTPUT_DIR),
                        data=os.path.relpath(data_save_path, OUTPUT_DIR),
                        num_frames=ed_idx-st_idx
                    ))

                    if os.path.exists(video_save_path) and os.path.exists(data_save_path):
                        with h5py.File(data_save_path, 'r+') as h5f:
                            # delete world_map to save space
                            if 'world_map' in h5f:
                                del h5f['world_map']
                            if 'camera_pose' in h5f:
                                print(f"Skip {video_save_path} and {data_save_path} already exists.")
                                continue

                    frames = []
                    disps = []
                    point_maps = []
                    valid_masks = []
                    camera_poses = []

                    for idx, (rgb_path, depth_path) in enumerate(zip(rgb_paths[st_idx:ed_idx], depth_paths[st_idx:ed_idx])):
                        
                        img = Image.open(os.path.join(DATA_DIR, rgb_path))
                        img = np.array(img).astype(np.uint8)
                        # grayscale images
                        if len(img.shape) == 2:
                            img = np.tile(img[..., None], (1, 1, 3))
                        else:
                            img = img[..., :3]

                        depth = np.load(os.path.join(DATA_DIR, depth_path))
                        invalid_mask = np.logical_or(depth > 1000, depth < DEPTH_EPS)

                        line = lines[st_idx + idx]
                        tx, ty, tz, qx, qy, qz, qw = [float(x) for x in line.split()]
                        camera_pose = ned_to_standard(tx, ty, tz, qx, qy, qz, qw)

                        depth[invalid_mask] = DEPTH_EPS
                        disp = 80.0 / depth
                        disp[invalid_mask] = 0.

                        valid_mask = np.logical_not(invalid_mask)  

                        fx = fy = 320.0
                        cx = 320
                        cy = 240

                        if img.shape[0] % 16 != 0  or img.shape[1] % 16 != 0:
                            H = img.shape[0] // 16 * 16
                            W = img.shape[1] // 16 * 16
                            x0, y0 = (img.shape[1] - W) // 2, (img.shape[0] - H) // 2
                            img = img[y0:y0+H, x0:x0+W, :]
                            depth = depth[y0:y0+H, x0:x0+W]
                            disp = disp[y0:y0+H, x0:x0+W]
                            valid_mask = valid_mask[y0:y0+H, x0:x0+W]
                            cx -= x0
                            cy -= y0

                        focal_length = np.array([fx, fy])
                        principal_point = np.array([cx, cy])
                        point_map = depth2point_map(
                            torch.tensor(depth).float().cuda(),
                            focal_length[0],
                            focal_length[1],
                            principal_point[0],
                            principal_point[1],
                        )
                        
                        frames.append(img)
                        disps.append(disp)
                        valid_masks.append(valid_mask)
                        point_maps.append(point_map.cpu().numpy())
                        camera_poses.append(camera_pose)

                        progress_bar.update(1)
                    
                    frames = np.stack(frames)
                    disps = np.stack(disps)
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