"""Preprocess Kubric sequences and optional dense tracking into clips."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
import glob
from tqdm import tqdm
import imageio
from preprocess_common import get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/Kubric/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/Kubric_video/")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 18)


def uv2point_map(uv, depth, fx, fy, cx, cy):
    device, dtype = uv.device, uv.dtype
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ], device=device, dtype=dtype)

    uv1 = torch.cat([uv, torch.ones_like(uv[..., :1])], dim=-1)  # (...,3)
    dirs = uv1 @ torch.inverse(K).T                             # (...,3)
    rays = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    points = rays * depth[..., None]
    return points

if __name__ == '__main__':
    
    meta_infos = []

    seq_names = sorted(os.listdir(DATA_DIR))

    for idx, seq_name in enumerate(seq_names):
        if seq_name.endswith('.txt'):
            continue
        rgb_paths = sorted(glob.glob(os.path.join(DATA_DIR, seq_name, 'video_frames', '*.png')))
        camera_path = os.path.join(DATA_DIR, seq_name, '{}_camera.npy'.format(seq_name))
        camera_poses = np.load(camera_path, allow_pickle=True).item()['camera2world_matrix']
            
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
            camera_poses = []
            scene_flows = []
            valid_masks = []
            visible_masks = []

            for idx in range(st_idx, ed_idx):
                rgb_path = rgb_paths[idx]
                scene_flow_path = os.path.join(DATA_DIR, seq_name, '{}_dense_tracking_{}.npy'.format(seq_name, str(idx // 2 * 2)))

                img = Image.open(rgb_path)
                img = np.array(img).astype(np.uint8)
                # grayscale images
                if len(img.shape) == 2:
                    img = np.tile(img[..., None], (1, 1, 3))
                else:
                    img = img[..., :3]

                focal_length = np.array([560, 560])
                principal_point = np.array([256, 256])
                camera_pose = camera_poses[idx]  # camera to world
                A_h = np.diag([1, -1, -1, 1])
                camera_pose = camera_pose @ A_h # to opengl

                try:
                    dense_tracking = np.load(scene_flow_path, allow_pickle=True).item()
                except (FileNotFoundError, OSError, ValueError):
                    print(f"Skip {scene_flow_path} not exists.")
                    record = False
                    break
                coords = dense_tracking['coords'][idx%2]
                order = np.lexsort((np.around(coords[:, idx, 0] + 0.5), np.around(coords[:, idx, 1] + 0.5)))
                coords_sorted = coords[order]
                depths = dense_tracking['reproj_depth'][idx%2]
                depth_sorted = depths[order]
                visibility = dense_tracking['visibility'][idx%2]
                visibility_sorted = visibility[order]
                depth = depth_sorted[:, idx].reshape(512, 512)
                valid_mask = depth > DEPTH_EPS

                uv = coords_sorted[:, idx].reshape(512, 512, 2) + 0.5
                point_map = uv2point_map(torch.tensor(uv).float().cuda(), 
                                         torch.tensor(depth).float().cuda(), 
                                         focal_length[0], focal_length[1], 
                                         principal_point[0], principal_point[1])

                if idx < ed_idx - 1:
                    deform_uv = coords_sorted[:, idx+1].reshape(512, 512, 2) + 0.5
                    forward_depth = depth_sorted[:, idx+1].reshape(512, 512)
                    forward_depth_mask = ~visibility_sorted[:, idx+1].reshape(512, 512)
                    point_map_deform = uv2point_map(torch.tensor(deform_uv).float().cuda(), 
                                                    torch.tensor(forward_depth).float().cuda(), 
                                                    focal_length[0], focal_length[1], 
                                                    principal_point[0], principal_point[1])
                    scene_flow = point_map_deform - point_map  # h,w,3
                    scene_flow[forward_depth_mask] = 0.0
                    visible_mask = forward_depth_mask
                else:
                    scene_flow = torch.zeros_like(point_map)
                    visible_mask = np.zeros_like(valid_mask)

                frames.append(img)
                valid_masks.append(valid_mask)
                point_maps.append(point_map.cpu().numpy())
                camera_poses.append(camera_pose)
                scene_flows.append(scene_flow.cpu().numpy())
                visible_masks.append(visible_mask)

                progress_bar.update(1)
            
            if record is False:
                continue
            
            os.makedirs(os.path.join(OUTPUT_DIR, seq_name), exist_ok=True)

            frames = np.stack(frames)
            valid_masks = np.stack(valid_masks)
            point_maps = np.stack(point_maps)
            camera_poses = np.stack(camera_poses)
            scene_flows = np.stack(scene_flows)
            deform_masks = np.stack(visible_masks)

            imageio.mimsave(video_save_path, frames, fps=24, quality=9)
            with h5py.File(data_save_path, 'w') as h5f:
                h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
                h5f.create_dataset('camera_pose', data=camera_poses.astype(np.float16), chunks=(1, )+camera_poses.shape[1:])
                h5f.create_dataset('scene_flow', data=scene_flows.astype(np.float16), chunks=(1, )+scene_flows.shape[1:])
                h5f.create_dataset('deform_mask', data=deform_masks.astype(np.bool_), chunks=(1, )+deform_masks.shape[1:])            
            break
    
    write_meta_infos(OUTPUT_DIR, meta_infos)