"""Preprocess Dynamic Replica sequences into MotionCrafter-compatible clips."""

import numpy as np
import h5py
import os
from PIL import Image
import torch
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from tqdm import tqdm
from collections import defaultdict
import imageio
from dataclasses import dataclass
from pytorch3d.implicitron.dataset.types import FrameAnnotation, load_dataclass
from typing import List, Optional
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.utils import opencv_from_cameras_projection
import gzip
from preprocess_common import depth2point_map, get_env_int, get_env_str, write_meta_infos

DATA_DIR = get_env_str("MOTIONCRAFTER_DATA_DIR", "data/datasets/dynamic_replica/")
OUTPUT_DIR = get_env_str("MOTIONCRAFTER_OUTPUT_DIR", "data/unnormed_datasets/dynamic_replica_video/")
SPLITS = ['train', 'valid']
DEPTH_EPS = 1e-5
CLIP_LENGTH = get_env_int("MOTIONCRAFTER_CLIP_LENGTH", 150)

@dataclass
class DynamicReplicaFrameAnnotation(FrameAnnotation):
    """A dataclass used to load annotations from json."""
    camera_name: Optional[str] = None
    # instance_id_map_path: Optional[str] = None
    flow_forward: Optional[str] = None
    flow_forward_mask: Optional[str] = None
    flow_backward: Optional[str] = None
    flow_backward_mask: Optional[str] = None
    trajectories: Optional[str] = None

def read_gen(file_name, pil=False):
    ext = os.path.splitext(file_name)[-1]
    if ext == ".png" or ext == ".jpeg" or ext == ".ppm" or ext == ".jpg":
        return Image.open(file_name)
    elif ext == ".bin" or ext == ".raw":
        return np.load(file_name)
    else:
        assert False

def depth2disparity_scale(left_camera, right_camera, image_size_tensor):
    # # opencv camera matrices
    (_, T1, K1), (_, T2, _) = [
        opencv_from_cameras_projection(
            f,
            image_size_tensor,
        )
        for f in (left_camera, right_camera)
    ]
    fix_baseline = T1[0][0] - T2[0][0]
    focal_length_px = K1[0][0][0]
    # following this https://github.com/princeton-vl/RAFT-Stereo#converting-disparity-to-depth
    return focal_length_px * fix_baseline

def load_16big_png_depth(filename):
    with Image.open(filename) as depth_pil:
        # the image is stored with 16-bit depth but PIL reads it as I (32 bit).
        # we cast it to uint16, then reinterpret as float16, then cast to float32
        depth = (
            np.frombuffer(np.array(depth_pil, dtype=np.uint16), dtype=np.float16)
            .astype(np.float32)
            .reshape((depth_pil.size[1], depth_pil.size[0]))
        )
    return depth


def get_pytorch3d_camera(
        entry_viewpoint, image_size, scale: float
    ) -> PerspectiveCameras:
    assert entry_viewpoint is not None
    # principal point and focal length
    principal_point = torch.tensor(
        entry_viewpoint.principal_point, dtype=torch.float
    )
    focal_length = torch.tensor(entry_viewpoint.focal_length, dtype=torch.float)

    half_image_size_wh_orig = (
        torch.tensor(list(reversed(image_size)), dtype=torch.float) / 2.0
    )

    # first, we convert from the dataset's NDC convention to pixels
    format = entry_viewpoint.intrinsics_format
    if format.lower() == "ndc_norm_image_bounds":
        # this is e.g. currently used in CO3D for storing intrinsics
        rescale = half_image_size_wh_orig
    elif format.lower() == "ndc_isotropic":
        rescale = half_image_size_wh_orig.min()
    else:
        raise ValueError(f"Unknown intrinsics format: {format}")

    # principal point and focal length in pixels
    principal_point_px = half_image_size_wh_orig - principal_point * rescale
    focal_length_px = focal_length * rescale

    # now, convert from pixels to PyTorch3D v0.5+ NDC convention
    # if self.image_height is None or self.image_width is None:
    out_size = list(reversed(image_size))

    half_image_size_output = torch.tensor(out_size, dtype=torch.float) / 2.0
    half_min_image_size_output = half_image_size_output.min()

    # rescaled principal point and focal length in ndc
    principal_point = (
        half_image_size_output - principal_point_px * scale
    ) / half_min_image_size_output
    focal_length = focal_length_px * scale / half_min_image_size_output

    return PerspectiveCameras(
        focal_length=focal_length[None],
        principal_point=principal_point[None],
        R=torch.tensor(entry_viewpoint.R, dtype=torch.float)[None],
        T=torch.tensor(entry_viewpoint.T, dtype=torch.float)[None],
    )

def get_output_tensor(sample, progress_bar):
    output_tensor = defaultdict(list)
    sample_size = len(sample["image"]["left"])
    output_tensor_keys = ["point_map", "valid_mask", "disparity", "depth", "focal_length", 
                          "principal_point", "img", "camera_pose", "scene_flow", "deform_mask"]

    for key in output_tensor_keys:
        output_tensor[key] = [None for _ in range(sample_size)]

    depth_eps = 1e-5

    for i in range(sample_size):
        # Load camera parameters.
        viewpoint_left = get_pytorch3d_camera(
            sample["viewpoint"]["left"][i],
            sample["metadata"]["left"][i][1],
            scale=1.0,
        )
        viewpoint_right = get_pytorch3d_camera(
            sample["viewpoint"]["right"][i],
            sample["metadata"]["right"][i][1],
            scale=1.0,
        )

        next_viewpoint_left = get_pytorch3d_camera(
            sample["viewpoint"]["left"][i+1] if i < sample_size - 1 else sample["viewpoint"]["left"][i],
            sample["metadata"]["left"][i][1],
            scale=1.0,
        )

        H, W = sample["metadata"]["left"][i][1]

        R, T, K = opencv_from_cameras_projection(
            viewpoint_left,
            torch.tensor([H, W])[None],
        )

        R_next, T_next, K_next = opencv_from_cameras_projection(
            next_viewpoint_left,
            torch.tensor([H, W])[None],
        )

        # load camera params
        camera_pose_left = torch.eye(4, device=R.device)
        camera_pose_left[:3,:3] = R[0].T
        camera_pose_left[:3,3] = -R[0].T @ T[0]

        camera_pose_left_next = torch.eye(4, device=R.device)
        camera_pose_left_next[:3,:3] = R_next[0].T
        camera_pose_left_next[:3,3] = -R_next[0].T @ T_next[0]

        output_tensor["camera_pose"][i] = camera_pose_left

        depth2disp_scale = depth2disparity_scale(
            viewpoint_left,
            viewpoint_right,
            torch.Tensor(sample["metadata"]["left"][i][1])[None],
        ).item()

        img = read_gen(sample["image"]['left'][i])
        img = np.array(img).astype(np.uint8)

        # grayscale images
        if len(img.shape) == 2:
            img = np.tile(img[..., None], (1, 1, 3))
        else:
            img = img[..., :3]
        output_tensor["img"][i] = img

        
        depth = load_16big_png_depth(sample["depth"]["left"][i])
        depth_mask = depth < depth_eps
        depth[depth_mask] = depth_eps
        output_tensor["depth"][i] = depth

        disp = depth2disp_scale / depth
        disp[depth_mask] = 0
        valid_mask = np.logical_and(disp < 512, np.logical_not(depth_mask))

        disp = np.array(disp).astype(np.float32)
        output_tensor["disparity"][i] = disp
        output_tensor["valid_mask"][i] = valid_mask

        pose_t = camera_pose_left
        if i < sample_size - 1:
            pose_t1 = camera_pose_left_next

        traj = torch.load(sample["trajectories"]['left'][i])
        traj_3d_w = traj["traj_3d_world"]  # (N,3)
        x_coords, y_coords = traj["traj_2d"][:,0], traj["traj_2d"][:,1]
        visibs = traj["verts_inds_vis"].numpy()

        if i < sample_size - 1:
            next_traj = torch.load(sample["trajectories"]['left'][i+1])
            next_traj_3d_w = next_traj["traj_3d_world"]
            # Convert to camera coordinates.
            def world_to_cam(p_world, pose):
                R = pose[:3, :3]
                t = pose[:3, 3]
                return (R.T @ (p_world - t).T).T
                
            p_cam_t = world_to_cam(traj_3d_w, pose_t)
            p_cam_t1 = world_to_cam(next_traj_3d_w, pose_t1)

            scene_flow_traj = p_cam_t1 - p_cam_t  # (N,3)
        else:
            scene_flow_traj = torch.zeros_like(traj_3d_w)

        scene_flow_map = np.zeros((H, W, 3), dtype=np.float32)
        deform_mask = np.zeros((H, W), dtype=bool)

        x_coords_int = torch.clamp(torch.round(x_coords).long(), 0, W - 1)
        y_coords_int = torch.clamp(torch.round(y_coords).long(), 0, H - 1)

        for idx in range(len(x_coords_int)):
            if visibs[idx]:
                x = int(x_coords_int[idx])
                y = int(y_coords_int[idx])
                scene_flow_map[y, x, :] = scene_flow_traj[idx].cpu().numpy()
                deform_mask[y, x] = True

        output_tensor["scene_flow"][i] = scene_flow_map
        output_tensor["deform_mask"][i] = deform_mask

        focal_length_ndc = viewpoint_left.focal_length[0].numpy()
        
        s = min(W, H)
        focal_length = focal_length_ndc / 2.0 * s
        output_tensor["focal_length"][i] = focal_length
        principal_point_ndc = viewpoint_left.get_principal_point()[0].numpy()
        principal_point = principal_point_ndc / 2.0 * s + np.array([W / 2.0, H/2.0])
        output_tensor["principal_point"][i] = principal_point
        progress_bar.update(1)

    return output_tensor


if __name__ == '__main__':
    
    meta_infos = []

    for split in SPLITS:
        frame_annotations_file = f'frame_annotations_{split}.jgz'
        with gzip.open(os.path.join(DATA_DIR, split, frame_annotations_file), "rt", encoding="utf8") as zipfile:
            frame_annos_list = load_dataclass(
                zipfile, List[DynamicReplicaFrameAnnotation]
            )

        seq_annos_dict = defaultdict(lambda: defaultdict(list))
        for frame_anno in frame_annos_list:
            seq_annos_dict[frame_anno.sequence_name][frame_anno.camera_name].append(
                frame_anno
            )
        
        for i, seq_name in enumerate(seq_annos_dict.keys()):
            filenames = defaultdict(lambda: defaultdict(list))
            for cam in ["left", "right"]: # iterate [left, right] view
                for frame_anno in seq_annos_dict[seq_name][cam]: # iterate each frame
                    rgb_path = os.path.join(DATA_DIR, split, frame_anno.image.path)
                    depth_path = os.path.join(DATA_DIR, split, frame_anno.depth.path)
                    dynamic_mask_path = os.path.join(DATA_DIR, split, frame_anno.mask.path)
                    try:
                        flow_path = os.path.join(DATA_DIR, split, frame_anno.flow_forward["path"])
                        flow_mask_path = os.path.join(DATA_DIR, split, frame_anno.flow_forward_mask["path"])
                    except (TypeError, KeyError):
                        flow_path = None
                        flow_mask_path = None
                    try:
                        traj_path = os.path.join(DATA_DIR, split, frame_anno.trajectories['path'])
                    except (TypeError, KeyError):
                        traj_path = None

                    assert os.path.isfile(rgb_path), rgb_path
                    assert os.path.isfile(depth_path), depth_path
                    
                    filenames["image"][cam].append(rgb_path)
                    filenames["depth"][cam].append(depth_path)

                    filenames["viewpoint"][cam].append(frame_anno.viewpoint)
                    filenames["metadata"][cam].append(
                        [frame_anno.sequence_name, frame_anno.image.size]
                    )

                    filenames["dynamic_mask"][cam].append(dynamic_mask_path)
                    filenames["flow_forward"][cam].append(flow_path)
                    filenames["flow_forward_mask"][cam].append(flow_mask_path)

                    filenames["trajectories"][cam].append(traj_path)

                    for k in filenames.keys():
                        assert (
                            len(filenames[k][cam])
                            == len(filenames["image"][cam])
                            > 0
                        ), frame_anno.sequence_name        

            os.makedirs(os.path.join(OUTPUT_DIR, split, seq_name), exist_ok=True)
            
            seq_len = len(filenames["image"]['left'])
            progress_bar = tqdm(
                range(seq_len),
            )
            progress_bar.set_description(f"Exec {seq_name} ({i}/{len(seq_annos_dict.keys())})")

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

                sample = defaultdict(lambda: defaultdict(list))
                for cam in ["left", "right"]:
                    for idx in range(st_idx, min(ed_idx, seq_len)):
                        for k in filenames.keys():
                            sample[k][cam].append(filenames[k][cam][idx])

                output_tensor = get_output_tensor(sample, progress_bar)
                
                depth = torch.from_numpy(np.stack(output_tensor["depth"], axis=0)).cuda()
                focal_length = torch.tensor(np.stack(output_tensor["focal_length"], axis=0)).cuda()
                principal_point = torch.tensor(np.stack(output_tensor["principal_point"], axis=0)).cuda()

                point_maps = depth2point_map(
                    depth, 
                    focal_length[:, 0], 
                    focal_length[:, 1], 
                    principal_point[:, 0], 
                    principal_point[:, 1],
                )


                point_maps = point_maps.detach().cpu().numpy().astype(np.float16)

                disps = np.array(output_tensor["disparity"]).astype(np.float16)
                valid_masks = np.array(output_tensor["valid_mask"]).astype(np.bool_)
                frames = np.array(output_tensor["img"]).astype(np.uint8)
                camera_poses = np.array(output_tensor["camera_pose"]).astype(np.float32)
                scene_flows = np.array(output_tensor["scene_flow"]).astype(np.float16)
                deform_masks = np.array(output_tensor["deform_mask"]).astype(np.bool_)
                
                imageio.mimsave(video_save_path, frames, fps=24, quality=9)
                with h5py.File(data_save_path, 'w') as h5f:
                    # h5f.create_dataset('disparity', data=disps, chunks=(1, )+disps.shape[1:])
                    h5f.create_dataset('valid_mask', data=valid_masks, chunks=(1, )+valid_masks.shape[1:])
                    h5f.create_dataset('point_map', data=point_maps, chunks=(1, )+point_maps.shape[1:])
                    # h5f.create_dataset('world_map', data=world_maps, chunks=(1, )+world_maps.shape[1:])
                    h5f.create_dataset('camera_pose', data=camera_poses, chunks=(1, )+camera_poses.shape[1:])
                    h5f.create_dataset('scene_flow', data=scene_flows, chunks=(1, )+scene_flows.shape[1:])
                    h5f.create_dataset('deform_mask', data=deform_masks, chunks=(1, )+deform_masks.shape[1:])


    write_meta_infos(OUTPUT_DIR, meta_infos)