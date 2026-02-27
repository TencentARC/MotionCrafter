"""Preprocess MatrixCity splits into MotionCrafter-ready shards."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
import math
from tqdm import tqdm
import imageio
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from preprocess_common import depth2point_map, get_env_int, get_env_list, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/MatrixCity/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/MatrixCity_video/")
SPLITS = get_env_list("MOTIONCRAFTER_SPLITS", "big_city_street_train,big_city_aerial_train,small_city_street_train,small_city_aerial_train")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def load_depth(depth_exr, is_float16=True):
    image = cv2.imread(depth_exr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[...,0] #(H, W)
    if is_float16==True:
        invalid_mask=(image>=65504)
    else:
        invalid_mask=np.zeros_like(image).astype(np.bool_)
    image = image / 10000 # cm -> 100m
    return image, invalid_mask

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

if __name__ == '__main__':
    
    meta_infos = []

    for split in SPLITS:
        
        if split == "big_city_street_train":
            scene_paths = [
                "big_city/street/train/bottom_area",
                "big_city/street/train/left_area",
                "big_city/street/train/right_area",
                "big_city/street/train/top_area"
            ]
            depth_paths = [
                "big_city_depth/street/train/bottom_area_depth",
                "big_city_depth/street/train/left_area_depth",
                "big_city_depth/street/train/right_area_depth",
                "big_city_depth/street/train/top_area_depth"
            ]
        elif split == "big_city_aerial_train":
            scene_paths = [
                "big_city/aerial/train/big_high_block_1",
                "big_city/aerial/train/big_high_block_2",
                "big_city/aerial/train/big_high_block_3",
                "big_city/aerial/train/big_high_block_4",
                "big_city/aerial/train/big_high_block_5",
                "big_city/aerial/train/big_high_block_6",
            ]
            depth_paths = [
                "big_city_depth/aerial/train/big_high_block_1_depth",
                "big_city_depth/aerial/train/big_high_block_2_depth",
                "big_city_depth/aerial/train/big_high_block_3_depth",
                "big_city_depth/aerial/train/big_high_block_4_depth",
                "big_city_depth/aerial/train/big_high_block_5_depth",
                "big_city_depth/aerial/train/big_high_block_6_depth",
            ]
        elif split == "small_city_street_train":
            scene_paths = [
                "small_city/street/train_dense/small_city_road_down_dense",
                "small_city/street/train_dense/small_city_road_horizon_dense",
                "small_city/street/train_dense/small_city_road_outside_dense",
                "small_city/street/train_dense/small_city_road_vertical_dense"
            ]
            depth_paths = [
                "small_city_depth/street/train_dense/small_city_road_down_dense_depth",
                "small_city_depth/street/train_dense/small_city_road_horizon_dense_depth",
                "small_city_depth/street/train_dense/small_city_road_outside_dense_depth",
                "small_city_depth/street/train_dense/small_city_road_vertical_dense_depth"
            ]
        elif split == "small_city_aerial_train":
            scene_paths = [
                "small_city/aerial/train/block_1", 
                "small_city/aerial/train/block_2",
                "small_city/aerial/train/block_3",
                "small_city/aerial/train/block_4", 
                "small_city/aerial/train/block_5",
                "small_city/aerial/train/block_6",
                "small_city/aerial/train/block_7", 
                "small_city/aerial/train/block_8",
                "small_city/aerial/train/block_9",
                "small_city/aerial/train/block_10",
            ]
            depth_paths = [
                "small_city_depth/aerial/train/block_1_depth", 
                "small_city_depth/aerial/train/block_2_depth",
                "small_city_depth/aerial/train/block_3_depth",
                "small_city_depth/aerial/train/block_4_depth", 
                "small_city_depth/aerial/train/block_5_depth",
                "small_city_depth/aerial/train/block_6_depth",
                "small_city_depth/aerial/train/block_7_depth", 
                "small_city_depth/aerial/train/block_8_depth",
                "small_city_depth/aerial/train/block_9_depth",
                "small_city_depth/aerial/train/block_10_depth",
            ]

        for idx, (scene_path, dep_path) in enumerate(zip(scene_paths, depth_paths)):
            os.makedirs(os.path.join(OUTPUT_DIR, split, scene_path), exist_ok=True)

            # load all images
            all_img_names = os.listdir(os.path.join(DATA_DIR, scene_path))
            all_img_names = [x for x in all_img_names if x.endswith(".png")]

            # for not zero padding image name
            all_img_names.sort()
            all_img_names = sorted(all_img_names, key=lambda x: int(x.split('.')[0]))  
            
            all_depth_names = [img_name.replace('.png', '.exr') for img_name in all_img_names]

            try:
                with open(os.path.join(DATA_DIR, scene_path, 'transforms_origin.json'), 'r') as ft:
                    transforms = json.load(ft)
            except FileNotFoundError:
                with open(os.path.join(DATA_DIR, scene_path, 'transforms.json'), 'r') as ft:
                    transforms = json.load(ft)

            camera_angle_x = transforms['camera_angle_x']        
            assert len(transforms['frames']) == len(all_img_names), \
                "Number of frames in transforms.json does not match number of images. Path: {}".format(os.path.join(DATA_DIR, scene_path, 'transforms.json'))

            seq_len = len(all_img_names)
            progress_bar = tqdm(
                range(seq_len),
            )
            progress_bar.set_description(f"Exec {scene_path} ({idx}/{len(scene_paths)})")

            for st_idx in range(0, seq_len, CLIP_LENGTH):
                ed_idx = st_idx + CLIP_LENGTH
                ed_idx = min(ed_idx, seq_len)
                video_save_path = os.path.join(OUTPUT_DIR, split, scene_path, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
                data_save_path = os.path.join(OUTPUT_DIR, split, scene_path, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

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

                for idx, (img_name, depth_name) in enumerate(zip(all_img_names[st_idx:ed_idx], all_depth_names[st_idx:ed_idx])):
                    rgb_path = os.path.join(scene_path, img_name)
                    depth_path = os.path.join(dep_path, depth_name)
                    img = Image.open(os.path.join(DATA_DIR, rgb_path))
                    img = np.array(img).astype(np.uint8)
                    # grayscale images
                    if len(img.shape) == 2:
                        img = np.tile(img[..., None], (1, 1, 3))
                    else:
                        img = img[..., :3]

                    depth, invalid_mask = load_depth(os.path.join(DATA_DIR, depth_path))
                    invalid_mask = np.logical_or(invalid_mask, depth < DEPTH_EPS)
                    depth[invalid_mask] = DEPTH_EPS
                    disp = 1.0 / depth
                    disp[invalid_mask] = 0.

                    valid_mask = np.logical_not(invalid_mask)  
                    # assert depth.shape == (1000, 1000)

                    focal_length = fov2focal(float(camera_angle_x), depth.shape[-1])
                    principal_point = np.array(list(reversed(depth.shape[-2:]))) / 2

                    fx = fy = focal_length
                    cx, cy = principal_point

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
                    camera_pose = np.array(transforms['frames'][st_idx+idx]['rot_mat'])
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
                    h5f.create_dataset('camera_pose', data=np.stack(camera_poses).astype(np.float16), chunks=(1, )+camera_poses.shape[1:])
    
    write_meta_infos(OUTPUT_DIR, meta_infos)