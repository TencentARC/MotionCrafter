"""Preprocess Point Odyssey into MotionCrafter clip-level data."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
import glob
from tqdm import tqdm
import imageio
import cv2
from preprocess_common import get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/pointodyssey")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/Point_odyssey_video/")
SPLIT = get_env_str("MOTIONCRAFTER_SPLIT", "val")  # 'train' or 'val'
process_scene_flow_map = get_env_str("MOTIONCRAFTER_PROCESS_SCENE_FLOW", "false").lower() in {"1", "true", "yes", "y"}
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

def load_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    invalid_mask = depth > 65500
    depth = depth / 65535.0 * 1000.0
    return depth, invalid_mask


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

def reprojection(points, K, RT, eps=1e-8):
    v = np.hstack([points, np.ones((points.shape[0], 1))])  # (N,4)
    XYZ = (RT @ v.T).T[:, :3]
    point_camera = XYZ.copy()
    # Z = XYZ[:, 2:]
    XYZ = XYZ / (XYZ[:, 2:]+eps)
    xyz = (K @ XYZ.T).T
    uv = xyz[:, :2]
    return point_camera, uv

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

    seq_names = sorted(os.listdir(os.path.join(DATA_DIR, SPLIT)))

    for idx, seq_name in enumerate(seq_names):
        if seq_name.endswith('.mp4') or seq_name.endswith('.txt') or seq_name.endswith('.py'):
            continue
        rgb_paths = sorted(glob.glob(os.path.join(DATA_DIR, SPLIT, seq_name, 'rgbs', '*.jpg')))
        depth_paths = [p.replace('rgbs', 'depths').replace('rgb', 'depth').replace('.jpg', '.png') for p in rgb_paths]
        npz_path = os.path.join(DATA_DIR, SPLIT, seq_name, 'anno.npz')
        annotations = np.load(npz_path, allow_pickle=True)
        intrinsics = annotations['intrinsics'].astype(np.float32)
        extrinsics = annotations['extrinsics'].astype(np.float32)
        traj_3d = annotations['trajs_3d']
        visibs = annotations['visibs']

        if traj_3d.shape[0] != len(rgb_paths):
            print(f"Skip {seq_name} due to inconsistent length between traj and rgb frames.")
            continue

        seq_len = len(rgb_paths)
        progress_bar = tqdm(
            range(seq_len),
        )
        progress_bar.set_description(f"Exec {seq_name} ({idx}/{len(seq_names)})")

        for st_idx in range(0, seq_len, CLIP_LENGTH):
            ed_idx = st_idx + CLIP_LENGTH
            ed_idx = min(ed_idx, seq_len)
            video_save_path = os.path.join(OUTPUT_DIR, seq_name, "{:05d}_{:05d}_rgb.mp4".format(st_idx, ed_idx))
            data_save_path = os.path.join(OUTPUT_DIR, seq_name, "{:05d}_{:05d}_data.hdf5".format(st_idx, ed_idx))

            frames = []
            point_maps = []
            camera_poses = []
            scene_flows = []
            valid_masks = []
            deform_masks = []

            try:
                for idx in range(st_idx, ed_idx):

                    rgb_path = rgb_paths[idx]
                    depth_path = depth_paths[idx]

                    img = Image.open(rgb_path)
                    img = np.array(img).astype(np.uint8)
                    H, W = img.shape[:2]
                    depth, invalid_mask = load_depth(depth_path)
                    valid_mask = ~invalid_mask & (depth > DEPTH_EPS)

                    camera_pose = extrinsics[idx]  # world to camera
                    intrinsic = intrinsics[idx]
                    if idx + 1 < seq_len:
                        camera_pose_next = extrinsics[idx+1] 
                    
                    point_map = uv2point_map(
                        uv=torch.tensor(np.stack(np.meshgrid(
                            np.arange(W), np.arange(H)
                        ), axis=-1).reshape(-1, 2), dtype=torch.float32).cuda(),
                        depth=torch.tensor(depth.reshape(-1), dtype=torch.float32).cuda(),
                        fx=intrinsic[0,0],
                        fy=intrinsic[1,1],
                        cx=intrinsic[0,2],
                        cy=intrinsic[1,2],
                    ).reshape(H, W, 3)  # (H,W,3)

                    
                    if idx + 1 < seq_len and process_scene_flow_map == True:
                        traj_3d_w = traj_3d[idx]  # (N,3)
                        traj_3d_w_next = traj_3d[idx+1]
                        visib = visibs[idx]     # (N,)
                        traj_3d_c, traj_2d = reprojection(
                            traj_3d_w,
                            K=intrinsic,
                            RT=camera_pose,
                        ) # (N,2)
                        traj_3d_c_next, _ = reprojection(
                            traj_3d_w_next,
                            K=intrinsic,
                            RT=camera_pose_next,
                        ) # (N,2)
                        scene_flow_traj = traj_3d_c_next - traj_3d_c  # (N,3)
                        scene_flow_map = np.zeros((H, W, 3), dtype=np.float32)
                        deform_mask = np.zeros((H, W), dtype=bool)

                        x_coords_int = np.clip(np.round(traj_2d[:,0]).astype(np.int64), 0, W - 1)
                        y_coords_int = np.clip(np.round(traj_2d[:,1]).astype(np.int64), 0, H - 1)

                        for i in range(len(x_coords_int)):
                            if visib[i]:
                                x = int(x_coords_int[i])
                                y = int(y_coords_int[i])
                                if 0 <= x < W and 0 <= y < H:
                                    scene_flow_map[y, x, :] = scene_flow_traj[i]
                                    deform_mask[y, x] = True
                        
                    else:
                        scene_flow_map = np.zeros((H, W, 3), dtype=np.float32)
                        deform_mask = np.zeros((H, W), dtype=bool)
                    

                    deform_mask = deform_mask & valid_mask

                    frames.append(img)
                    valid_masks.append(valid_mask)
                    point_maps.append(point_map.cpu().numpy())
                    camera_poses.append(np.linalg.inv(camera_pose))  # camera to world
                    scene_flows.append(scene_flow_map)
                    deform_masks.append(deform_mask)
                        
                    progress_bar.update(1)
                    
            except Exception as e:
                print(f"Error processing {seq_name} frames {st_idx}-{ed_idx}: {e}")
                continue

            meta_infos.append(dict(
                video=os.path.relpath(video_save_path, OUTPUT_DIR),
                data=os.path.relpath(data_save_path, OUTPUT_DIR),
                num_frames=ed_idx-st_idx
            ))
            os.makedirs(os.path.join(OUTPUT_DIR, seq_name), exist_ok=True)

            frames = np.stack(frames)
            valid_masks = np.stack(valid_masks)
            point_maps = np.stack(point_maps)
            camera_poses = np.stack(camera_poses)
            scene_flows = np.stack(scene_flows)
            deform_masks = np.stack(deform_masks)

            h, w = frames.shape[1:3]
            h_crop = h // 16 * 16
            w_crop = w // 16 * 16
            if h_crop != h or w_crop != w:
                frames = center_crop(frames, (h_crop, w_crop))
                valid_masks = center_crop(valid_masks, (h_crop, w_crop))
                point_maps = center_crop(point_maps, (h_crop, w_crop))
                scene_flows = center_crop(scene_flows, (h_crop, w_crop))
                deform_masks = center_crop(deform_masks, (h_crop, w_crop))


            imageio.mimsave(video_save_path, frames, fps=24, quality=9)
            with h5py.File(data_save_path, 'w') as h5f:
                h5f.create_dataset('valid_mask', data=valid_masks.astype(np.bool_), chunks=(1, )+valid_masks.shape[1:])
                h5f.create_dataset('point_map', data=point_maps.astype(np.float16), chunks=(1, )+point_maps.shape[1:])
                h5f.create_dataset('camera_pose', data=camera_poses.astype(np.float16), chunks=(1, )+camera_poses.shape[1:])
                if process_scene_flow_map == True:
                    h5f.create_dataset('scene_flow', data=scene_flows.astype(np.float16), chunks=(1, )+scene_flows.shape[1:])
                    h5f.create_dataset('deform_mask', data=deform_masks.astype(np.bool_), chunks=(1, )+deform_masks.shape[1:])            
    
    write_meta_infos(OUTPUT_DIR, meta_infos)