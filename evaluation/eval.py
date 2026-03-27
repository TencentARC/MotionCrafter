import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

try:
    from .metrics import (
        depth_inlier_percent,
        depth_rel_error,
        point_inlier_percent,
        point_rel_error,
        project_to_depth_map,
        sceneflow_metrics,
    )
except ImportError:
    from metrics import (
        depth_inlier_percent,
        depth_rel_error,
        point_inlier_percent,
        point_rel_error,
        project_to_depth_map,
        sceneflow_metrics,
    )


EVAL_METRICS = [
    "point_abs_relative_difference",
    "point_delta1_acc",
    "depth_abs_relative_difference",
    "depth_delta1_acc",
    "scene_flow_epe",
    "scene_flow_acc_003",
    "scene_flow_acc_005",
    "scene_flow_acc_01",
    "scene_flow_acc_03",
]


def resolve_device(device_name: str) -> torch.device:
    # Keep a single device resolver so CLI and internal calls stay consistent.
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, please use --device cpu or --device auto")
    return torch.device(device_name)


def recover_scale_shift(points, points_gt, mask=None, weight=None):
    """Recover global scale and shift so that points_gt ~= scale * points + shift."""
    assert points.shape[-1] == 3, "This version assumes 3D points."
    device = points.device

    # Flatten all frames/pixels into a single point set for robust global alignment.
    points = points.reshape(-1, 3)
    points_gt = points_gt.reshape(-1, 3)

    if mask is not None:
        mask = mask.reshape(-1)
        points = points[mask]
        points_gt = points_gt[mask]
        if weight is not None:
            weight = weight.reshape(-1)[mask]
    elif weight is not None:
        weight = weight.reshape(-1)

    if weight is None:
        weight = torch.ones(points.shape[0], device=device)
    else:
        weight = weight.to(device)

    # Weighted Procrustes-style fit (scale + translation only, no rotation).
    w_sum = torch.clamp_min(torch.sum(weight), 1e-12)
    mean_p = torch.sum(points * weight[:, None], dim=0) / w_sum
    mean_p_gt = torch.sum(points_gt * weight[:, None], dim=0) / w_sum

    p_centered = points - mean_p
    p_gt_centered = points_gt - mean_p_gt

    numerator = torch.sum(weight * torch.sum(p_gt_centered * p_centered, dim=1))
    denominator = torch.clamp_min(torch.sum(weight * torch.sum(p_centered**2, dim=1)), 1e-12)
    scale = numerator / denominator
    shift = mean_p_gt - scale * mean_p
    return scale, shift


def normalize_pose_to_first(pose: torch.Tensor) -> torch.Tensor:
    # Normalize trajectory so frame-0 is identity to remove global gauge freedom.
    ref_inv = torch.linalg.inv(pose[0])
    return ref_inv[None, :, :] @ pose


def to_world(point_map: torch.Tensor, pose_c2w: torch.Tensor, device: torch.device) -> torch.Tensor:
    # point_map is in camera coordinates; pose_c2w maps camera points into world.
    t, h, w, _ = point_map.shape
    point_map = point_map.reshape(t, h * w, 3)
    point_map_h = torch.cat([point_map, torch.ones((t, h * w, 1), device=device)], dim=-1)
    point_map_world = torch.bmm(point_map_h, pose_c2w.transpose(1, 2))[..., :3]
    return point_map_world.reshape(t, h, w, 3)


def resize_to_match(pred: torch.Tensor, target_hw) -> torch.Tensor:
    # Resize in CHW for interpolation, then return to HWC convention.
    if pred.shape[1:3] == target_hw:
        return pred
    pred = pred.permute(0, 3, 1, 2)
    pred = torch.nn.functional.interpolate(pred, size=target_hw, mode="bilinear", align_corners=True)
    return pred.permute(0, 2, 3, 1)


def load_samples(gt_data_dir: str, use_normed_data: bool):
    # Two metadata conventions are supported for backward compatibility.
    if use_normed_data:
        meta_file_path = os.path.join(gt_data_dir, "meta_infos.txt")
    else:
        meta_file_path = os.path.join(gt_data_dir, "filename_list.txt")

    if not os.path.exists(meta_file_path):
        raise FileNotFoundError(meta_file_path)

    samples = []
    with open(meta_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if use_normed_data:
                # meta_infos.txt has 3 columns: video_path data_path score_or_extra
                video_path, data_path, _ = line.split()
                data_path = data_path.replace("_data.hdf5", "_normed_data_320_640.hdf5")
            else:
                # filename_list.txt has 2 columns: video_path data_path
                video_path, data_path = line.split()

            samples.append({"video_path": video_path, "data_path": data_path})
    return samples


def eval_single(pred_path, gt_path, vggt_pose_path, args, device):
    # Prediction convention: point_map always required, scene_flow optional.
    pred_data = np.load(pred_path)
    pred_pmap = pred_data["point_map"].astype(np.float32)
    pred_sflow = pred_data["scene_flow"].astype(np.float32) if "scene_flow" in pred_data else None

    with h5py.File(gt_path, "r") as file:
        gt_mask = file["valid_mask"][:].astype(np.bool_)
        gt_pmap = file["point_map"][:].astype(np.float32)
        gt_pose = file["camera_pose"][:].astype(np.float32)
        gt_sflow = file["scene_flow"][:].astype(np.float32) if "scene_flow" in file else None
        gt_dmask = file["deform_mask"][:].astype(np.bool_) if "deform_mask" in file else None

    # Historical protocol: evaluate 25 frames for geometry-only and 8 for flow.
    if gt_sflow is None or pred_sflow is None:
        test_num_frames = min(args.max_frames_no_flow, gt_pmap.shape[0], pred_pmap.shape[0])
    else:
        test_num_frames = min(args.max_frames_with_flow, gt_sflow.shape[0], pred_sflow.shape[0], gt_pmap.shape[0], pred_pmap.shape[0])

    gt_mask = torch.from_numpy(gt_mask[:test_num_frames]).bool().to(device)
    gt_pmap = torch.from_numpy(gt_pmap[:test_num_frames]).float().to(device)
    gt_pose = torch.from_numpy(gt_pose[:test_num_frames]).float().to(device)
    gt_sflow = torch.from_numpy(gt_sflow[:test_num_frames]).float().to(device) if gt_sflow is not None else None
    gt_dmask = torch.from_numpy(gt_dmask[:test_num_frames]).bool().to(device) if gt_dmask is not None else None
    gt_dmask = gt_dmask & gt_mask if gt_dmask is not None else gt_mask

    pred_pmap = torch.from_numpy(pred_pmap[:test_num_frames]).float().to(device)
    pred_sflow = torch.from_numpy(pred_sflow[:test_num_frames]).float().to(device) if pred_sflow is not None else None

    if pred_pmap.shape[0] != gt_pmap.shape[0]:
        raise ValueError(f"Frame number mismatch: pred {pred_pmap.shape[0]} vs gt {gt_pmap.shape[0]}")

    pred_pmap = resize_to_match(pred_pmap, (gt_pmap.shape[1], gt_pmap.shape[2]))
    if pred_sflow is not None:
        pred_sflow = resize_to_match(pred_sflow, (gt_pmap.shape[1], gt_pmap.shape[2]))

    weight_map = None
    if args.use_weight:
        # Heavier weight for near-depth points, matching prior evaluation setting.
        weight_map = (1.0 / (gt_pmap[..., 2] + 1e-6)) * gt_mask

    # Canonicalize GT trajectory before any coordinate conversion.
    gt_pose = normalize_pose_to_first(gt_pose)

    aligned_sflow = None
    gt_eval_pmap = gt_pmap

    if args.is_pred_world_map:
        # Predicted map is already in world coordinates: only align scale/shift to GT world.
        gt_eval_pmap = to_world(gt_pmap, gt_pose, device=device)
        scale, shift = recover_scale_shift(pred_pmap, gt_eval_pmap, mask=gt_mask, weight=weight_map)
        aligned_pmap = pred_pmap * scale + shift

        if pred_sflow is not None and gt_sflow is not None:
            # World-space flow under pure scale/shift alignment only scales by scale.
            aligned_sflow = pred_sflow * scale

        if args.save_aligned_world:
            # Optional debug artifact for downstream visualization/inspection.
            np.savez(
                pred_path[:-4] + "_aligned_world.npz",
                point_map=aligned_pmap.detach().cpu().numpy().astype(np.float16),
                scene_flow=aligned_sflow.detach().cpu().numpy().astype(np.float16) if aligned_sflow is not None else None,
                valid_mask=gt_mask.detach().cpu().numpy().astype(np.bool_),
            )
    else:
        # Predicted map is camera-space: align in camera space first, then transform to world.
        scale, shift = recover_scale_shift(pred_pmap, gt_pmap, mask=gt_mask, weight=weight_map)
        aligned_pmap_cam = pred_pmap * scale + shift

        gt_eval_pmap = to_world(gt_pmap, gt_pose, device=device)

        if args.use_vggt_pose:
            # Replace GT pose with external pose estimate (e.g., VGGT) for world transform.
            if not os.path.exists(vggt_pose_path):
                raise FileNotFoundError(f"VGGT pose file not found: {vggt_pose_path}")
            vggt_pose = np.load(vggt_pose_path)["camera_pose"][:test_num_frames]
            vggt_pose = torch.from_numpy(vggt_pose).float().to(device)
            vggt_pose = normalize_pose_to_first(vggt_pose)
            aligned_pmap = to_world(aligned_pmap_cam, vggt_pose, device=device)

            if pred_sflow is not None and gt_sflow is not None:
                # Compose deformed points in camera space, then convert each frame to world.
                aligned_pmap_deformed_cam = (pred_pmap + pred_sflow) * scale + shift
                next_vggt_pose = vggt_pose if args.static_pose_for_flow else torch.roll(vggt_pose, shifts=-1, dims=0)
                aligned_pmap_deformed = to_world(aligned_pmap_deformed_cam, next_vggt_pose, device=device)
                aligned_sflow = aligned_pmap_deformed - aligned_pmap
        else:
            # Default branch: use GT pose for world conversion.
            aligned_pmap = to_world(aligned_pmap_cam, gt_pose, device=device)

            if pred_sflow is not None and gt_sflow is not None:
                aligned_pmap_deformed_cam = (pred_pmap + pred_sflow) * scale + shift
                next_gt_pose = gt_pose if args.static_pose_for_flow else torch.roll(gt_pose, shifts=-1, dims=0)
                aligned_pmap_deformed = to_world(aligned_pmap_deformed_cam, next_gt_pose, device=device)
                aligned_sflow = aligned_pmap_deformed - aligned_pmap

    # Geometry metrics in world coordinates.
    p_rel_err = point_rel_error(aligned_pmap, gt_eval_pmap, gt_mask).item()
    p_in_percent = point_inlier_percent(aligned_pmap, gt_eval_pmap, gt_mask).item()

    # Depth metrics are computed by projecting world points back to camera depth.
    aligned_dmap = project_to_depth_map(aligned_pmap, gt_pose)
    gt_dmap = project_to_depth_map(gt_eval_pmap, gt_pose)

    d_rel_err = depth_rel_error(aligned_dmap, gt_dmap, gt_mask).item()
    d_in_percent = depth_inlier_percent(aligned_dmap, gt_dmap, gt_mask).item()

    if aligned_sflow is not None and gt_sflow is not None:
        # Use T-1 valid flow pairs; the last frame has no forward target.
        sflow = sceneflow_metrics(aligned_sflow[:-1], gt_sflow[:-1], gt_dmask[:-1])
        sflow_metrics_list = [m.item() for m in sflow]
    else:
        # Keep fixed output shape when flow annotations/predictions are unavailable.
        sflow_metrics_list = [-1.0, -1.0, -1.0, -1.0, -1.0]

    return [p_rel_err, p_in_percent, d_rel_err, d_in_percent] + sflow_metrics_list


def parse_args():
    parser = argparse.ArgumentParser(description="MotionCrafter evaluation script")
    parser.add_argument("--pred_data_dir", type=str, required=True, help="Predicted output directory")
    parser.add_argument("--gt_data_dir", type=str, required=True, help="GT dataset directory")
    parser.add_argument("--vggt_pose_dir", type=str, default="", help="Directory with VGGT pose npz files")
    parser.add_argument("--is_pred_world_map", action="store_true", help="Predicted point map is already in world coordinates")
    parser.add_argument("--use_weight", action="store_true", help="Use depth-based weight map during alignment")
    parser.add_argument("--use_vggt_pose", action="store_true", help="Use VGGT pose for world transformation")
    parser.add_argument("--use_normed_data", action="store_true", help="Read meta_infos.txt and _normed_data_320_640.hdf5")
    parser.add_argument("--save_file_name", type=str, default="metrics.json", help="Output json filename")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Device for evaluation")
    parser.add_argument("--max_frames_no_flow", type=int, default=25, help="Max frames when scene flow is unavailable")
    parser.add_argument("--max_frames_with_flow", type=int, default=8, help="Max frames when scene flow is available")
    parser.add_argument("--strict_missing", action="store_true", help="Raise error on missing input files")
    parser.add_argument("--save_aligned_world", action="store_true", help="Save aligned world-space predictions next to npz files")
    parser.add_argument("--static_pose_for_flow", action="store_true", help="Use same pose for flow transformation (for ST4RTrack/POMATO style data)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print(f"[Eval] Using device: {device}")

    # Build sample table from GT metadata list.
    samples = load_samples(args.gt_data_dir, args.use_normed_data)
    results_all = []
    evaluated_video_paths = []
    skipped = []

    for sample in tqdm(samples, desc="Evaluating"):
        gt_path = os.path.join(args.gt_data_dir, sample["data_path"])
        pred_path = os.path.join(args.pred_data_dir, sample["video_path"][:-4] + ".npz")
        vggt_pose_path = os.path.join(args.vggt_pose_dir, sample["video_path"][:-4] + "_pose_aligned.npz")

        missing = []
        if not os.path.exists(gt_path):
            missing.append(gt_path)
        if not os.path.exists(pred_path):
            missing.append(pred_path)
        if args.use_vggt_pose and not os.path.exists(vggt_pose_path):
            missing.append(vggt_pose_path)

        if missing:
            msg = f"Missing required files for {sample['video_path']}: {missing}"
            if args.strict_missing:
                raise FileNotFoundError(msg)
            print(f"[Skip] {msg}")
            skipped.append({"video_path": sample["video_path"], "missing": missing})
            continue

        results_single = eval_single(pred_path, gt_path, vggt_pose_path, args, device)
        results_all.append(results_single)
        evaluated_video_paths.append(sample["video_path"])

    if not results_all:
        raise RuntimeError("No valid samples were evaluated. Please check input directories.")

    # Aggregate per-sample metrics into final means.
    final_results = np.array(results_all, dtype=np.float64)
    final_results_mean = np.mean(final_results, axis=0)

    result_dict = {
        "_meta": {
            "num_samples_total": len(samples),
            "num_samples_evaluated": len(results_all),
            "num_samples_skipped": len(skipped),
            "device": str(device),
        }
    }

    for i, metric_name in enumerate(EVAL_METRICS):
        result_dict[metric_name] = float(final_results_mean[i])
        print(f"{metric_name}: {final_results_mean[i]:.6f}")

    for i, video_path in enumerate(evaluated_video_paths):
        result_dict[video_path] = results_all[i]

    if skipped:
        result_dict["_skipped"] = skipped

    # Ensure output directory exists even when only partial samples were evaluated.
    save_json_path = os.path.join(args.pred_data_dir, args.save_file_name)
    Path(args.pred_data_dir).mkdir(parents=True, exist_ok=True)
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=4)

    print(f"\nEvaluation results saved to: {save_json_path}")


if __name__ == "__main__":
    main()
    
