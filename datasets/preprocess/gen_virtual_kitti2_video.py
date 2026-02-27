"""Preprocess Virtual KITTI 2 sequences into MotionCrafter outputs."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
import glob
from tqdm import tqdm
import imageio
import cv2
import pandas as pd
from preprocess_common import depth2point_map, get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/Virtual_KITTI_2/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/Virtual_KITTI_2_video/")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def load_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    invalid_mask = depth > 65500
    depth = depth / 100.0 # cm -> m
    return depth, invalid_mask

def read_vkitti_scene_flow(flow_fn):
    # Convert from .png to (h, w, 2) (flow_x, flow_y) float32 array
    # read png to bgr in 16 bit unsigned short

    bgr = cv2.imread(flow_fn, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    h, w, _c = bgr.shape
    assert bgr.dtype == np.uint16 and _c == 3
    # b == invalid flow flag == 0 for sky or other invalid flow
    #   invalid = bgr[..., 0] == 0
    # g,r == flow_y,x normalized by height,width and scaled to [0;2**16 – 1]
    out_flow = (2.0 / (2**16 - 1.0) * bgr[..., ::-1].astype('f4') - 1) * 10.0
    #   out_flow[invalid] = 0 # or another value (e.g., np.nan)
    return out_flow

if __name__ == '__main__':
    
    meta_infos = []

    scene_names = ['Scene01','Scene02','Scene06','Scene18','Scene20']
    env_names = ['15-deg-left','15-deg-right','30-deg-left','30-deg-right','clone','fog','morning','overcast','rain','sunset']

    intrinsic_dict = dict()
    extrinsic_dict = dict()

    for scene_name in scene_names:
        for env_name in env_names:
            intr_path = os.path.join(DATA_DIR, scene_name, env_name, 'intrinsic.txt')
            intr_data = pd.read_csv(intr_path, sep="\s+")
            intr_data = pd.DataFrame(intr_data)
            for i, item in intr_data.iterrows():
                intrinsic_dict[f"{scene_name}_{env_name}_{int(item['frame'])}_{int(item['cameraID'])}"] = dict(
                    focal_length=(float(item['K[0,0]']), float(item['K[1,1]'])),
                    principal_point=(float(item['K[0,2]']), float(item['K[1,2]']))
                )

            entr_path = os.path.join(DATA_DIR, scene_name, env_name, 'extrinsic.txt')
            entr_data = pd.read_csv(entr_path, sep="\s+")
            entr_data = pd.DataFrame(entr_data)
            for i, item in entr_data.iterrows():
                extrinsic_dict[f"{scene_name}_{env_name}_{int(item['frame'])}_{int(item['cameraID'])}"] = np.array([
                    [float(item['r1,1']), float(item['r1,2']), float(item['r1,3']), float(item['t1'])],
                    [float(item['r2,1']), float(item['r2,2']), float(item['r2,3']), float(item['t2'])],
                    [float(item['r3,1']), float(item['r3,2']), float(item['r3,3']), float(item['t3'])],
                    [0.0, 0.0, 0.0, 1.0]
                ])

    splits = os.listdir(DATA_DIR)
    splits = sorted(list(filter(lambda x: not x.endswith('txt'), splits)))

    for split in splits:
        seq_names = sorted(os.listdir(os.path.join(DATA_DIR, split)))
        seq_names = filter(lambda x: not (
                x.endswith('.txt')) ,seq_names)
        seq_names = list(seq_names)

        for idx, seq_name in enumerate(seq_names):
            cams = sorted(os.listdir(os.path.join(DATA_DIR, split, seq_name, 'frames', 'rgb')))
            for cam in cams:
                rgb_paths = glob.glob(os.path.join(DATA_DIR, split, seq_name, 'frames', 'rgb', cam, 'rgb_*.jpg'))
                rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
                rgb_paths = sorted(rgb_paths, key=lambda p: int(os.path.basename(p).split('_')[-1][:-4]))
                depth_paths = [
                    p.replace('/rgb/', '/depth/').replace('/rgb_','/depth_').replace('.jpg', '.png')
                    for p in rgb_paths]
                
                os.makedirs(os.path.join(OUTPUT_DIR, split, seq_name, cam), exist_ok=True)
                
                seq_len = len(rgb_paths)
                progress_bar = tqdm(
                    range(seq_len),
                )
                progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

                for st_idx in range(0, seq_len, CLIP_LENGTH):
                    ed_idx = st_idx + CLIP_LENGTH
                    ed_idx = min(ed_idx, seq_len)
                    video_save_path = os.path.join(OUTPUT_DIR, split, seq_name, cam, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
                    data_save_path = os.path.join(OUTPUT_DIR, split, seq_name, cam, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

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
                    camera_poses = []
                    scene_flows = []
                    valid_masks = []

                    for idx in range(st_idx, ed_idx):
                        rgb_path = rgb_paths[idx]
                        depth_path = rgb_path.replace('/rgb/', '/depth/').replace('/rgb_','/depth_').replace('.jpg', '.png')
                        scene_flow_path = depth_path.replace('/depth/', '/forwardSceneFlow/').replace('/depth_','/sceneFlow_')

                        scene_name = rgb_path.split('/')[0]
                        env_name = rgb_path.split('/')[1]
                        frame = int(rgb_path.split('/')[-1][4:-4])
                        intrinsics = intrinsic_dict[f"{scene_name}_{env_name}_{frame}_{cam.split('_')[-1]}"]
                        camera_pose = extrinsic_dict[f"{scene_name}_{env_name}_{frame}_{cam.split('_')[-1]}"]
                        img = Image.open(os.path.join(DATA_DIR, rgb_path))
                        img = np.array(img).astype(np.uint8)
                        # grayscale images
                        if len(img.shape) == 2:
                            img = np.tile(img[..., None], (1, 1, 3))
                        else:
                            img = img[..., :3]

                        depth, invalid_mask = load_depth(os.path.join(DATA_DIR, depth_path))
                        invalid_mask = np.logical_or(depth < 1e-5, invalid_mask)
                        depth[invalid_mask] = DEPTH_EPS
                        disp = 1.0 / depth
                        disp[invalid_mask] = 0.

                        valid_mask = np.logical_not(invalid_mask)

                        if idx < ed_idx - 1:
                            scene_flow = read_vkitti_scene_flow(os.path.join(DATA_DIR, scene_flow_path))
                        else:
                            scene_flow = np.zeros_like(img)

                        focal_length = np.array(intrinsics['focal_length'])
                        principal_point = np.array(intrinsics['principal_point'])


                        fx, fy = focal_length[0], focal_length[1]
                        cx, cy = principal_point[0], principal_point[1]

                        if img.shape[0] % 16 != 0  or img.shape[1] % 16 != 0:
                            H = img.shape[0] // 16 * 16
                            W = img.shape[1] // 16 * 16
                            x0, y0 = (img.shape[1] - W) // 2, (img.shape[0] - H) // 2
                            img = img[y0:y0+H, x0:x0+W, :]
                            depth = depth[y0:y0+H, x0:x0+W]
                            disp = disp[y0:y0+H, x0:x0+W]
                            valid_mask = valid_mask[y0:y0+H, x0:x0+W]
                            scene_flow = scene_flow[y0:y0+H, x0:x0+W, :]
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
                        scene_flows.append(scene_flow)

                        progress_bar.update(1)
                    
                    frames = np.stack(frames)
                    disps = np.stack(disps)
                    valid_masks = np.stack(valid_masks)
                    point_maps = np.stack(point_maps)
                    camera_poses = np.stack(camera_poses)
                    scene_flows = np.stack(scene_flows)

                    imageio.mimsave(video_save_path, frames, fps=24, quality=9)
                    with h5py.File(data_save_path, 'w') as h5f:
                        # h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
                        h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                        h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
                        h5f.create_dataset('camera_pose', data=camera_poses.astype(np.float16), chunks=(1, )+camera_poses.shape[1:])
                        h5f.create_dataset('scene_flow', data=scene_flows.astype(np.float16), chunks=(1, )+scene_flows.shape[1:])
    
    write_meta_infos(OUTPUT_DIR, meta_infos)