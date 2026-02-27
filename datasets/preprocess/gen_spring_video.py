"""Preprocess Spring data into MotionCrafter clips with scene flow."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
import glob
from tqdm import tqdm
from kornia.geometry.depth import depth_to_3d_v2
from kornia.utils import create_meshgrid
import imageio
from preprocess_common import get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/Spring/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/Spring_video/")
SPLIT = get_env_str("MOTIONCRAFTER_SPLIT", "train")
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def readDsp5Disp(filename):
    with h5py.File(filename, "r") as f:
        if "disparity" not in f.keys():
            raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
        return f["disparity"][()]

def readFlo5Flow(filename):
    with h5py.File(filename, "r") as f:
        if "flow" not in f.keys():
            raise IOError(f"File {filename} does not have a 'flow' key. Is this a valid flo5 file?")
        return f["flow"][()]

def flow2scene_flow(flow, depth, depth_2, fx, fy, cx, cy):
    # [h,w,2] [h,w] [h,w] [1,] [1,] [1,] [1,]
    assert len(depth.shape) == 2 and len(flow.shape) == 3 and flow.shape[2] == 2
    camera_matrix = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ], dtype=depth.dtype, device=depth.device)
    point_map = depth_to_3d_v2(
        depth.unsqueeze(0),
        camera_matrix.unsqueeze(0),
        normalize_points=False
    ) # 1,h,w,3

    flow = flow.unsqueeze(0) # 1,h,w,2
    _,h,w,_ = flow.shape
    pixel_coords = create_meshgrid(h, w, normalized_coordinates=False).to(flow.device) # 1,h,w,2
    pixel_coords = pixel_coords + flow # 1,h,w,2

    homog_pixel_coords = torch.cat([
        pixel_coords,
        torch.ones_like(pixel_coords[..., :1])
    ], dim=-1) # 1,h,w,3

    cam_coords = torch.matmul(
        torch.inverse(camera_matrix).unsqueeze(0).unsqueeze(0).unsqueeze(0), # 1,1,1,3,3
        homog_pixel_coords.unsqueeze(-1), # 1,h,w,3,1
    ).squeeze(-1) * depth_2.unsqueeze(0).unsqueeze(-1).float() # 1,h,w,3

    scene_flow = cam_coords - point_map # 1,h,w,3

    return point_map.squeeze(0), scene_flow.squeeze(0)

if __name__ == '__main__':
    
    meta_infos = []
    seq_names = sorted(os.listdir(os.path.join(DATA_DIR, SPLIT)))

    for idx, seq_name in enumerate(seq_names):
        rgb_paths = glob.glob(os.path.join(DATA_DIR, SPLIT, seq_name, 'frame_left', '*.png'))
        rgb_paths = [os.path.relpath(p, DATA_DIR) for p in rgb_paths]
        rgb_paths = sorted(rgb_paths, key=lambda p: int(os.path.basename(p).split('_')[-1][:-4]))
        disp_paths = [p.replace('frame', 'disp1').replace('.png', '.dsp5') for p in rgb_paths]
        disp_2_paths = [p.replace('frame', 'disp2_FW').replace('.png', '.dsp5') for p in rgb_paths]
        flow_paths = [p.replace('frame', 'flow_FW').replace('.png', '.flo5') for p in rgb_paths]
        meta_paths = [os.path.join('/'.join(p.split('/')[:-2]), 'cam_data', 'intrinsics.txt') for p in rgb_paths]
        pose_paths = [os.path.join('/'.join(p.split('/')[:-2]), 'cam_data', 'extrinsics.txt') for p in rgb_paths]
        
        os.makedirs(os.path.join(OUTPUT_DIR, SPLIT, seq_name), exist_ok=True)
        
        seq_len = len(rgb_paths) - 1
        progress_bar = tqdm(
            range(seq_len),
        )
        progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

        for st_idx in range(0, seq_len, CLIP_LENGTH):
            ed_idx = st_idx + CLIP_LENGTH
            ed_idx = min(ed_idx, seq_len)
            video_save_path = os.path.join(OUTPUT_DIR, SPLIT, seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
            data_save_path = os.path.join(OUTPUT_DIR, SPLIT, seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

            meta_infos.append(dict(
                video=os.path.relpath(video_save_path, OUTPUT_DIR),
                data=os.path.relpath(data_save_path, OUTPUT_DIR),
                num_frames=ed_idx-st_idx
            ))

            frames = []
            disps = []
            point_maps = []
            world_maps = []
            camera_poses = []
            scene_flows = []
            deform_masks = []
            valid_masks = []

            for rgb_path, disp_path, disp_2_path, flow_path, meta_path, pose_path in zip(
                rgb_paths[st_idx:ed_idx], 
                disp_paths[st_idx:ed_idx], 
                disp_2_paths[st_idx:ed_idx],
                flow_paths[st_idx:ed_idx],
                meta_paths[st_idx:ed_idx], 
                pose_paths[st_idx:ed_idx]
            ):
                frame_index = int(os.path.basename(rgb_path).split('_')[-1][:-4])
                with open(os.path.join(DATA_DIR, meta_path), 'r') as f:
                    lines = f.readlines()
                    line = lines[frame_index-1]
                fx, fy, cx, cy = line.split()
                fx, fy, cx, cy=float(fx), float(fy), float(cx), float(cy)
                with open(os.path.join(DATA_DIR, pose_path), 'r') as f:
                    lines = f.readlines()
                    line = lines[frame_index-1]
                camera_pose = np.array([float(x) for x in line.split()]).reshape(4, 4)
                camera_pose = np.linalg.inv(camera_pose) # cam to world
                img = Image.open(os.path.join(DATA_DIR, rgb_path))
                img = np.array(img).astype(np.uint8)
                # grayscale images
                if len(img.shape) == 2:
                    img = np.tile(img[..., None], (1, 1, 3))
                else:
                    img = img[..., :3]

                disp = readDsp5Disp(os.path.join(DATA_DIR, disp_path))
                disp = disp[::2,::2]
                invalid_mask = np.isnan(disp) | (disp == 0)
                disp[invalid_mask] = 0.0
                depth = 0.065 * fx / np.clip(disp, 1e-2, disp.max())
                invalid_mask = invalid_mask | (depth < DEPTH_EPS)
                depth[invalid_mask] = DEPTH_EPS
                valid_mask = np.logical_not(invalid_mask)  

                disp_2 = readDsp5Disp(os.path.join(DATA_DIR, disp_2_path))
                disp_2 = disp_2[::2,::2]
                invalid_mask_2 = np.isnan(disp_2) | (disp_2 == 0)
                disp_2[invalid_mask_2] = 0.0
                depth_2 = 0.065 * fx / np.clip(disp_2, 1e-2, disp_2.max())
                invalid_mask_2 = invalid_mask_2 | (depth_2 < DEPTH_EPS)
                depth_2[invalid_mask_2] = DEPTH_EPS
                deform_mask = valid_mask & np.logical_not(invalid_mask_2)

                flow = readFlo5Flow(os.path.join(DATA_DIR, flow_path))
                flow = flow[::2,::2,:]
                flow = np.array(flow).astype(np.float32)[:,:,:2]
                flow_invalid_mask = np.isnan(flow).any(axis=2)
                flow[flow_invalid_mask] = 0.0
                deform_mask = deform_mask & np.logical_not(flow_invalid_mask)

                if img.shape[0] % 16 != 0  or img.shape[1] % 16 != 0:
                    H = img.shape[0] // 16 * 16
                    W = img.shape[1] // 16 * 16
                    x0, y0 = (img.shape[1] - W) // 2, (img.shape[0] - H) // 2
                    img = img[y0:y0+H, x0:x0+W, :]
                    depth = depth[y0:y0+H, x0:x0+W]
                    depth_2 = depth_2[y0:y0+H, x0:x0+W]
                    disp = disp[y0:y0+H, x0:x0+W]
                    flow = flow[y0:y0+H, x0:x0+W, :]
                    valid_mask = valid_mask[y0:y0+H, x0:x0+W]
                    deform_mask = deform_mask[y0:y0+H, x0:x0+W]
                    cx -= x0
                    cy -= y0

                focal_length = np.array([fx, fy])
                principal_point = np.array([cx, cy])

                point_map, scene_flow = flow2scene_flow(
                    torch.tensor(flow).float().cuda(), 
                    torch.tensor(depth).float().cuda(), 
                    torch.tensor(depth_2).float().cuda(), 
                    focal_length[0], focal_length[1], 
                    principal_point[0], principal_point[1]
                )

                frames.append(img)
                disps.append(disp)
                valid_masks.append(valid_mask)
                point_maps.append(point_map.cpu().numpy())
                camera_poses.append(camera_pose)
                scene_flows.append(scene_flow.cpu().numpy())
                deform_masks.append(deform_mask)

                progress_bar.update(1)
            
            frames = np.stack(frames)
            disps = np.stack(disps)
            valid_masks = np.stack(valid_masks)
            point_maps = np.stack(point_maps)
            camera_poses = np.stack(camera_poses)
            scene_flows = np.stack(scene_flows)
            deform_masks = np.stack(deform_masks)

            imageio.mimsave(video_save_path, frames, fps=24, quality=9)
            with h5py.File(data_save_path, 'w') as h5f:
                # h5f.create_dataset('disparity', data=disps.astype(np.float16), chunks=(1, )+disps.shape[1:])
                h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
                h5f.create_dataset('camera_pose', data=camera_poses.astype(np.float16), chunks=(1, )+camera_poses.shape[1:])
                h5f.create_dataset('scene_flow', data=scene_flows.astype(np.float16), chunks=(1, )+scene_flows.shape[1:])
                h5f.create_dataset('deform_mask', data=deform_masks.astype(np.bool_), chunks=(1, )+deform_masks.shape[1:])



    write_meta_infos(OUTPUT_DIR, meta_infos)