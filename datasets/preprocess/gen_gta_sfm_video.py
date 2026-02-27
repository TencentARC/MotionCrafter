"""Preprocess GTA-SfM h5 scenes into MotionCrafter video/hdf5 shards."""

import numpy as np
import h5py
import cv2
import os
import torch
from tqdm import tqdm
import imageio
from preprocess_common import depth2point_map, get_env_int, get_env_list, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/GTA-SfM/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/GTA-SfM_video/")
SPLITS = get_env_list("MOTIONCRAFTER_SPLITS", "train,test")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

if __name__ == '__main__':
    
    meta_infos = []

    for split in SPLITS:
        seq_names = sorted(os.listdir(os.path.join(DATA_DIR, split)))

        for idx, seq_name in enumerate(seq_names):
            samples = []
            h5f = h5py.File(os.path.join(DATA_DIR, split, seq_name), "r")
            num_frames = len(h5f.keys()) // 4
            for i in range(num_frames):
                samples.append(dict(
                    h5file=h5f,
                    img_name = "image_%d" % i,
                    K_name = "K_%d" % i,
                    pose_name = "pose_%d" % i,
                    depth_name = "depth_%d" % i,
                ))
            
            os.makedirs(os.path.join(OUTPUT_DIR, split, seq_name), exist_ok=True)
            
            seq_len = len(samples)
            progress_bar = tqdm(
                range(seq_len),
            )
            progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

            for st_idx in range(0, seq_len, CLIP_LENGTH):
                ed_idx = st_idx + CLIP_LENGTH
                ed_idx = min(ed_idx, seq_len)
                video_save_path = os.path.join(OUTPUT_DIR, split, seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
                data_save_path = os.path.join(OUTPUT_DIR, split, seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

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

                for sample in samples[st_idx:ed_idx]:
                    h5file = sample['h5file']
                    img = cv2.imdecode(h5file[sample['img_name']][:], cv2.IMREAD_COLOR)
                    img = np.array(img).astype(np.uint8)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]

                    K = h5file[sample['K_name']][:]
                    depth = h5file[sample['depth_name']][:]
                    camera_pose = h5file[sample['pose_name']][:]
                    focal_length_x, focal_length_y = K[0][0], K[1][1]
                    cx, cy = K[0][2], K[1][2]
                    invalid_mask = (depth < DEPTH_EPS) | (depth > 256.0)
                    depth[invalid_mask] = DEPTH_EPS
                    disp = 1.0 / depth
                    disp[invalid_mask] = 0.
                    valid_mask = np.logical_not(invalid_mask)  

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

                    focal_length = np.array([focal_length_x, focal_length_y])
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