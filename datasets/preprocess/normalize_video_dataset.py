"""Normalize generated preprocess clips to a target training resolution."""

import os
import os.path as osp
import argparse
import shutil
from typing import Union, Optional, Dict

import h5py
import imageio
import numpy as np
import torch
import torch.nn as nn
import torchvision
from decord import VideoReader, cpu
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torchvision.transforms.v2 import Compose, InterpolationMode, functional as F


torchvision.disable_beta_transforms_warning()

RESOLUTION = (320, 640)


def normalize_camera_pose(camera_pose: torch.Tensor) -> torch.Tensor:
    # camera_pose: T, 4, 4 (camera-to-world)
    first_pose = camera_pose[0].clone()
    camera_pose[:, :3, :3] = first_pose[:3, :3].T @ camera_pose[:, :3, :3]
    camera_pose[:, :3, 3] = (
        first_pose[:3, :3].T @ (camera_pose[:, :3, 3] - first_pose[:3, 3]).T
    ).T
    return camera_pose.clone()


def infer_output_dir(data_dir: str) -> str:
    if "unnormed_datasets" in data_dir:
        return data_dir.replace("unnormed_datasets", "tmp_datasets")
    return os.path.join(data_dir, "normalized")


class CoverResize(nn.Module):
    def __init__(
        self,
        interpolation: Union[InterpolationMode, int] = InterpolationMode.NEAREST,
        antialias: Optional[Union[str, bool]] = False,
    ):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias

    def _get_params(self, size, cropped_size):
        ratio = max(cropped_size[0] / size[0], cropped_size[1] / size[1])
        return int(size[0] * ratio), int(size[1] * ratio)

    def forward(self, input_dict: Dict) -> Dict:
        frame = input_dict["frame"]
        size = frame.shape[-2:]
        cropped_size = input_dict.get("resolution") or size
        new_size = self._get_params(size, cropped_size)

        need_resize = new_size[0] != size[0] or new_size[1] != size[1]
        if not need_resize:
            return input_dict

        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor) and value.shape[-2:] == size:
                output_dict[key] = F.resize(
                    value,
                    new_size,
                    interpolation=self.interpolation,
                    antialias=self.antialias,
                )
            else:
                output_dict[key] = value
        return output_dict


class CenterCrop(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_dict: Dict) -> Dict:
        frame = input_dict["frame"]
        size = frame.shape[-2:]
        cropped_size = input_dict.get("resolution") or size
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

        needs_crop = needs_vert_crop or needs_horz_crop
        if not needs_crop:
            return input_dict

        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, torch.Tensor) and value.shape[-2:] == size:
                output_dict[key] = F.crop(
                    value,
                    top=top,
                    left=left,
                    height=cropped_height,
                    width=cropped_width,
                )
            else:
                output_dict[key] = value

        return output_dict


class Video(Dataset):
    def __init__(
        self,
        data_dir,
        output_dir=None,
        resolution=RESOLUTION,
        skip_existing=False,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir is not None else infer_output_dir(data_dir)
        self.resolution = tuple(resolution)
        self.skip_existing = skip_existing
        os.makedirs(self.output_dir, exist_ok=True)

        meta_file_path = os.path.join(data_dir, "meta_infos.txt")
        assert os.path.exists(meta_file_path), meta_file_path

        shutil.copy(meta_file_path, osp.join(self.output_dir, "meta_infos.txt"))

        self.samples = []
        with open(meta_file_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                video_path, data_path, _ = line.split()
                self.samples.append({"video_path": video_path, "data_path": data_path})

        transforms = [CoverResize(antialias=True), CenterCrop()]
        self.transform = Compose(transforms)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        video_path = sample["video_path"]
        data_path = sample["data_path"]

        normalized_video_path = video_path.replace(
            "_rgb.mp4", f"_rgb_{self.resolution[0]}_{self.resolution[1]}.mp4"
        )
        normalized_data_path = data_path.replace(
            "_data.hdf5", f"_normed_data_{self.resolution[0]}_{self.resolution[1]}.hdf5"
        )

        if self.skip_existing and os.path.exists(
            osp.join(self.output_dir, normalized_video_path)
        ) and os.path.exists(osp.join(self.output_dir, normalized_data_path)):
            return 0

        try:
            sample_np = self._get_output_numpy(sample)
        except Exception as exc:
            print(exc)
            print(data_path)
            return 0

        sample_tensor = self._to_float_tensor(sample_np)

        frame = sample_tensor.pop("frame") / 255.0
        valid_mask = sample_tensor.pop("valid_mask")
        point_map = sample_tensor.pop("point_map")
        camera_pose = sample_tensor.pop("camera_pose", None)
        scene_flow = sample_tensor.pop("scene_flow", None)
        deform_mask = sample_tensor.pop("deform_mask", None)

        data = {
            "frame": frame.permute(0, 3, 1, 2),
            "valid_mask": valid_mask,
            "point_map": point_map.permute(0, 3, 1, 2),
            "camera_pose": camera_pose,
            "scene_flow": scene_flow.permute(0, 3, 1, 2) if scene_flow is not None else None,
            "deform_mask": deform_mask,
            "resolution": self.resolution,
        }

        data = self.transform(data)
        data.pop("resolution")

        frame = data.pop("frame").permute(0, 2, 3, 1)
        frame = (frame * 255).numpy().astype(np.uint8)

        os.makedirs(osp.dirname(osp.join(self.output_dir, normalized_video_path)), exist_ok=True)
        imageio.mimsave(osp.join(self.output_dir, normalized_video_path), frame, fps=24, quality=9)

        valid_mask = data.pop("valid_mask")
        point_map = data.pop("point_map")
        camera_pose = data.pop("camera_pose", None)
        scene_flow = data.pop("scene_flow", None)
        deform_mask = data.pop("deform_mask", None)

        point_map = point_map.permute(0, 2, 3, 1)
        norm_factor = 1.0
        point_map = point_map.numpy()

        if camera_pose is not None:
            camera_pose[:, :3, 3] = camera_pose[:, :3, 3] / norm_factor

            if "matrixcity" in self.data_dir.lower():
                camera_pose[:, :3, :3] = camera_pose[:, :3, :3] * 100
                camera_pose[:, :3, 3] = camera_pose[:, :3, 3] / 100
                camera_pose[:, :3, 2] = -camera_pose[:, :3, 2]
                camera_pose[:, 2, 3] = -camera_pose[:, 2, 3]

            if "mvs-synth" in self.data_dir.lower() or "virtual_kitti_2" in self.data_dir.lower():
                camera_pose = torch.inverse(camera_pose)

            if "tartanair" in self.data_dir.lower():
                camera_pose[:, :, :3] = camera_pose[:, :, [1, 2, 0]].clone()

            camera_pose = normalize_camera_pose(camera_pose)
            camera_pose = camera_pose.numpy()

        if scene_flow is not None:
            scene_flow = scene_flow.permute(0, 2, 3, 1)
            scene_flow = (scene_flow / norm_factor).numpy()

        valid_mask = (valid_mask > 0.9).numpy()
        deform_mask = (deform_mask > 0.9).numpy() & valid_mask if deform_mask is not None else None

        with h5py.File(osp.join(self.output_dir, normalized_data_path), "w") as h5f:
            h5f.create_dataset(
                "valid_mask", data=valid_mask.astype(np.bool_), chunks=(1,) + valid_mask.shape[1:]
            )
            h5f.create_dataset(
                "point_map", data=point_map.astype(np.float16), chunks=(1,) + point_map.shape[1:]
            )

            if camera_pose is not None:
                h5f.create_dataset(
                    "camera_pose",
                    data=camera_pose.astype(np.float16),
                    chunks=(1,) + camera_pose.shape[1:],
                )

            if scene_flow is not None:
                h5f.create_dataset(
                    "scene_flow",
                    data=scene_flow.astype(np.float16),
                    chunks=(1,) + scene_flow.shape[1:],
                )

            if deform_mask is not None:
                h5f.create_dataset(
                    "deform_mask",
                    data=deform_mask.astype(np.bool_),
                    chunks=(1,) + deform_mask.shape[1:],
                )

        return 0

    def _to_float_tensor(self, sample):
        output_tensor = {}
        for key, value in sample.items():
            output_tensor[key] = None if value is None else torch.tensor(value).float()
        return output_tensor

    def _get_output_numpy(self, sample):
        video_reader = VideoReader(osp.join(self.data_dir, sample["video_path"]), ctx=cpu(0))

        with h5py.File(osp.join(self.data_dir, sample["data_path"]), "r") as file:
            valid_masks = file["valid_mask"][:]
            point_maps = file["point_map"][:]
            camera_poses = file["camera_pose"][:]

            try:
                scene_flows = file["scene_flow"][:]  # optional
            except KeyError:
                scene_flows = None

            try:
                deform_mask = file["deform_mask"][:]  # optional
            except KeyError:
                try:
                    deform_mask = ~file["visible_mask"][:]  # legacy optional key
                except KeyError:
                    deform_mask = None

            if deform_mask is None and scene_flows is not None:
                deform_mask = np.ones_like(valid_masks, dtype=np.bool_)

        frame = video_reader.get_batch(list(range(len(video_reader))))

        return {
            "frame": frame.asnumpy(),  # T,H,W,3
            "point_map": point_maps,  # T,H,W,3
            "valid_mask": valid_masks,  # T,H,W
            "camera_pose": camera_poses,  # T,4,4
            "scene_flow": scene_flows,  # T,H,W,3
            "deform_mask": deform_mask,  # T,H,W
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize preprocessed videos and hdf5 annotations.")
    parser.add_argument(
        "--data-dirs",
        nargs="+",
        required=True,
        help="Input dataset directories containing meta_infos.txt",
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=None,
        help="Output directories aligned with --data-dirs",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output root; per-dataset dirname will be appended",
    )
    parser.add_argument("--resolution", nargs=2, type=int, default=list(RESOLUTION), metavar=("H", "W"))
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.output_dirs is not None and len(args.output_dirs) != len(args.data_dirs):
        raise ValueError("--output-dirs and --data-dirs must have the same length")

    resolved_output_dirs = []
    for idx, data_dir in enumerate(args.data_dirs):
        if args.output_dirs is not None:
            resolved_output_dirs.append(args.output_dirs[idx])
        elif args.output_root is not None:
            resolved_output_dirs.append(
                os.path.join(args.output_root, os.path.basename(os.path.normpath(data_dir)))
            )
        else:
            resolved_output_dirs.append(infer_output_dir(data_dir))

    for data_dir, output_dir in zip(args.data_dirs, resolved_output_dirs):
        print(f"normalize {data_dir} -> {output_dir}")
        dataset = Video(
            data_dir=data_dir,
            output_dir=output_dir,
            resolution=tuple(args.resolution),
            skip_existing=args.skip_existing,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=False,
        )
        for _ in tqdm(dataloader, total=len(dataset)):
            pass
