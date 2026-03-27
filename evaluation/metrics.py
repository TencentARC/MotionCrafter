import torch


def abs_relative_difference(output, target, valid_mask=None):
    # Mean over frames of: |pred-gt| / gt, aggregated on spatial dims.
    actual_output = output
    actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = torch.sum(abs_relative_diff, (-1, -2)) / n
    return abs_relative_diff.mean()


def squared_relative_difference(output, target, valid_mask=None):
    # Mean over frames of: |pred-gt|^2 / gt, aggregated on spatial dims.
    actual_output = output
    actual_target = target
    square_relative_diff = (
        torch.pow(torch.abs(actual_output - actual_target), 2) / actual_target
    )
    if valid_mask is not None:
        square_relative_diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = torch.sum(square_relative_diff, (-1, -2)) / n
    return square_relative_diff.mean()


def rmse_linear(output, target, valid_mask=None):
    # Standard RMSE in linear domain.
    actual_output = output
    actual_target = target
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n
    rmse = torch.sqrt(mse)
    return rmse.mean()


def rmse_log(output, target, valid_mask=None):
    # RMSE in log-depth domain.
    diff = torch.log(output) - torch.log(target)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def log10(output, target, valid_mask=None):
    # Mean absolute log10 difference.
    if valid_mask is not None:
        diff = torch.abs(
            torch.log10(output[valid_mask]) - torch.log10(target[valid_mask])
        )
    else:
        diff = torch.abs(torch.log10(output) - torch.log10(target))
    return diff.mean()


# adapt from: https://github.com/imran3180/depth-map-prediction/blob/master/main.py
def threshold_percentage(output, target, threshold_val, valid_mask=None):
    # Delta accuracy: percentage of pixels where max(pred/gt, gt/pred) < threshold.
    d1 = output / target
    d2 = target / output
    max_d1_d2 = torch.max(d1, d2)
    zero = torch.zeros(*output.shape)
    one = torch.ones(*output.shape)
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    if valid_mask is not None:
        bit_mat[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = torch.sum(bit_mat, (-1, -2))
    threshold_mat = count_mat / n.cpu()
    return threshold_mat.mean()


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    # RMSE in inverse-depth domain.
    output_inv = 1.0 / output
    target_inv = 1.0 / target
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (-1, -2)) / n  # [B]
    rmse = torch.sqrt(mse)
    return rmse.mean()


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    # Scale-invariant log RMSE, multiplied by 100 (common reporting convention).
    diff = torch.log(depth_pred) - torch.log(depth_gt)
    if valid_mask is not None:
        diff[~valid_mask] = 0
        n = valid_mask.sum((-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = torch.pow(diff, 2)

    first_term = torch.sum(diff2, (-1, -2)) / n
    second_term = torch.pow(torch.sum(diff, (-1, -2)), 2) / (n**2)
    loss = torch.sqrt(torch.mean(first_term - second_term)) * 100
    return loss


def point_rel_error(point_map, point_map_gt, mask_gt):
    assert point_map.shape == point_map_gt.shape, "{} != {}".format(point_map.shape, point_map_gt.shape)
    assert point_map_gt.shape[:-1] == mask_gt.shape, "{} != {}".format(point_map_gt.shape[:-1], mask_gt.shape)
    # point_map / point_map_gt: [*, H, W, 3], mask_gt: [*, H, W]
    # Relative 3D endpoint error normalized by GT point norm.
    error = torch.norm(point_map - point_map_gt, p=2, dim=-1, keepdim=False)
    rel_error = error / torch.clamp_min(torch.norm(point_map_gt, p=2, dim=-1, keepdim=False), 1e-2)
    rel_error = (rel_error * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return rel_error.mean()

def depth_rel_error(depth, depth_gt, mask_gt):
    assert depth.shape == depth_gt.shape, "{} != {}".format(depth.shape, depth_gt.shape)
    assert depth_gt.shape == mask_gt.shape, "{} != {}".format(depth_gt.shape, mask_gt.shape)

    # Relative absolute depth error: |pred-gt| / |gt|.
    error = (depth - depth_gt).abs()
    rel_error = error / torch.clamp_min(depth_gt.abs(), 1e-2)
    rel_error = (rel_error * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return rel_error.mean()

def point_inlier_percent(point_map, point_map_gt, mask_gt):
    assert point_map.shape == point_map_gt.shape
    assert point_map_gt.shape[:-1] == mask_gt.shape
    # Inlier if relative 3D error < 0.25.
    error = torch.norm(point_map - point_map_gt, p=2, dim=-1, keepdim=False)
    rel_error = error / torch.clamp_min(torch.norm(point_map_gt, p=2, dim=-1, keepdim=False), 1e-2)
    percentage = ((rel_error < 0.25).float() * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return percentage.mean()

def depth_inlier_percent(depth, depth_gt, mask_gt):
    assert depth.shape == depth_gt.shape
    assert depth_gt.shape == mask_gt.shape
    # Inlier if depth ratio max(pred/gt, gt/pred) < 1.25.
    error = torch.max(depth.abs()/torch.clamp_min(depth_gt.abs(), 1e-2), depth_gt.abs()/torch.clamp_min(depth.abs(), 1e-2))
    percentage = ((error < 1.25).float() * mask_gt.float()).sum((-1, -2)) / mask_gt.float().sum((-1, -2))
    return percentage.mean()

def project_to_depth_map(point_map, camera_pose):
    """
    point_map: [B', H, W, 3]
    camera_pose: [B', 4, 4]
    return: depth_map: [B', H, W]
    """
    B_shape = point_map.shape[:-3]
    H = point_map.shape[-3]
    W = point_map.shape[-2]
    

    point_map_flat = point_map.reshape(-1, H*W, 3).transpose(1, 2)  # [B', 3, H*W]
    B_flat = point_map_flat.shape[0]
    if camera_pose is not None:
        camera_pose_flat = camera_pose.reshape(-1, 4, 4)  # [B', 4, 4]
        assert camera_pose_flat.shape[0] == B_flat
        R = camera_pose_flat[:, :3, :3]  # [B', 3, 3]
        t = camera_pose_flat[:, :3, 3:]  # [B', 3, 1]
        # Convert world points to camera frame and read z as depth.
        point_map_cam_flat = torch.bmm(R.transpose(1, 2), (point_map_flat - t))  # [B', 3, H*W]
        point_map_cam_flat = point_map_cam_flat.transpose(1, 2)  # [B', H*W, 3]
    else:
        print("Warning: camera_pose is None in project_to_depth_map!")
        point_map_cam_flat = point_map_flat
    depth_map_flat = point_map_cam_flat[:, :, 2]  # [B', H*W]
    depth_map = depth_map_flat.reshape(*B_shape, H, W)  # [*, H, W]
    return depth_map


def sceneflow_metrics(pred_flow, gt_flow, valid_mask=None):
    """
    pred_flow: [B, T, H, W, 3]
    gt_flow: [B, T, H, W, 3]
    valid_mask: [B, T, H, W]
    """
    # Endpoint error per pixel in 3D flow space.
    error = torch.norm(pred_flow - gt_flow, p=2, dim=-1, keepdim=False)  # [B, T, H, W]

    if valid_mask is not None:
        error = error * valid_mask.float()
        n = valid_mask.sum((-1, -2, -3))
    else:
        n = pred_flow.shape[-1] * pred_flow.shape[-2] * pred_flow.shape[-3]
    epe = torch.sum(error, (-1, -2, -3)) / n
    epe = epe.mean()
    # APD@x is implemented as 1 - outlier_rate(error > x).
    apd_003 = 1 - (torch.sum((error > 0.03).float(), (-1, -2, -3)) / n)
    apd_003 = apd_003.mean()
    apd_005 = 1 - (torch.sum((error > 0.05).float(), (-1, -2, -3)) / n)
    apd_005 = apd_005.mean()
    apd_01 = 1 - (torch.sum((error > 0.1).float(), (-1, -2, -3)) / n)
    apd_01 = apd_01.mean()
    apd_03 = 1 - (torch.sum((error > 0.3).float(), (-1, -2, -3)) / n)
    apd_03 = apd_03.mean()

    return epe, apd_003, apd_005, apd_01, apd_03