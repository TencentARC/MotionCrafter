"""Interactive viser-based visualizer for point clouds, camera poses, and scene flow."""

import time
from pathlib import Path
import numpy as np
import tyro
from tqdm.auto import tqdm
import h5py
import viser
import viser.extras
import viser.transforms as tf
from decord import VideoReader, cpu
import os.path as osp
import argparse
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt


def point_map_edge(point_map, dist_threshold=0.05):
    """
    Detect edge regions in a world-coordinate point map.

    Args:
        point_map: numpy array [H, W, 3], world-space coordinates
        dist_threshold: float, Euclidean distance threshold to consider a pixel an edge

    Returns:
        edge_mask: boolean array [H, W], True indicates an edge pixel
    """
    diff_x = np.linalg.norm(point_map[:, 1:] - point_map[:, :-1], axis=-1)
    diff_y = np.linalg.norm(point_map[1:] - point_map[:-1], axis=-1)

    diff_x = np.pad(diff_x, ((0, 0), (0, 1)), mode="edge")
    diff_y = np.pad(diff_y, ((0, 1), (0, 0)), mode="edge")

    edge_mask = (diff_x > dist_threshold) | (diff_y > dist_threshold)
    return edge_mask


def main(
    downsample_factor: int = 1,
    max_frames: int = 100,
    share: bool = False,
) -> None:
    """Launch the viewer to explore RGB video, point maps, and optional scene flow overlays."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the data file containing point maps and masks",
    )
    args = parser.parse_args()

    if not osp.exists(args.video_path):
        raise FileNotFoundError(f"Video file {args.video_path} does not exist.")

    # Auto-detect data file path
    args.data_path = args.data_path or args.video_path.replace("rgb.mp4", "rec.hdf5")
    if not osp.exists(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} does not exist.")

    # Launch visualization server
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading frames!")

    # Load video & 3D reconstruction data
    video_reader = VideoReader(args.video_path, ctx=cpu(0))

    if args.data_path.endswith(".hdf5"):
        with h5py.File(args.data_path, "r") as file:
            # print("Data keys:", list(file.keys()))

            point_maps = file["point_map"][:]  # (t, h, w, 3)
            # print(
            #     "max depth:", point_maps[..., 2].max(),
            #     "min depth:", point_maps[..., 2].min()
            # )

            try:
                valid_masks = file["valid_mask"][:]
            except:
                valid_masks = np.ones_like(point_maps[..., 0]).astype(bool)

            try:
                camera_poses = file["camera_pose"][:]  # (t, 4, 4)
            except:
                camera_poses = np.tile(np.eye(4)[None], (valid_masks.shape[0], 1, 1))
                print("No camera pose found, using identity.")

            try:
                scene_flows = file["scene_flow"][:]
            except:
                scene_flows = None

            try:
                deform_masks = file["deform_mask"][:]
            except:
                deform_masks = np.ones_like(valid_masks).astype(bool)

    elif args.data_path.endswith(".npz"):
        data = np.load(args.data_path)

        point_maps = data["point_map"]
        # print(point_maps.shape)

        valid_masks = data.get("valid_mask", np.ones_like(point_maps[..., 0]).astype(bool))
        try:
            camera_poses = data["camera_pose"]
        except:
            camera_poses = np.tile(np.eye(4)[None], (valid_masks.shape[0], 1, 1))
            print("No camera pose found, using identity.")
        scene_flows = data.get("scene_flow", None)
        deform_masks = data.get("deform_mask", np.ones_like(valid_masks).astype(bool))

    else:
        raise ValueError("Unsupported data file format. Use .hdf5 or .npz.")

    num_frames = min(max_frames, len(video_reader))
    # num_frames = min(num_frames, 7)
    print(f"Visualizing {num_frames} frames.")

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        """Initialize default camera pose when a client connects."""
        client.camera.position = (-1.554, -1.013, 1.142)
        client.camera.look_at = (-0.005, 2.283, -0.156)

    # GUI control panel
    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size", min=0.001, max=0.1, step=0.005, initial_value=0.01
        )
        gui_line_size = server.gui.add_slider(
            "Line size", min=0.1, max=10.0, step=0.1, initial_value=1.0
        )
        gui_timestep = server.gui.add_slider(
            "Timestep", min=0, max=num_frames - 1, step=1, initial_value=0, disabled=True
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1,
            initial_value=round(video_reader.get_avg_fps())
        )

    # GUI callbacks
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    prev_timestep = gui_timestep.value

    @gui_timestep.on_update
    def _(_) -> None:
        """Toggle frame and point cloud visibility when timestep changes."""
        nonlocal prev_timestep
        current_timestep = gui_timestep.value

        with server.atomic():
            frame_nodes[current_timestep].visible = False
            frame_nodes[prev_timestep].visible = False

            point_static_nodes[current_timestep].visible = True
            point_static_nodes[prev_timestep].visible = False
            point_dynamic_nodes[current_timestep].visible = True
            point_dynamic_nodes[prev_timestep].visible = False

            if len(point_deform_nodes) > 0:
                point_deform_nodes[prev_timestep].visible = False
                scene_flow_nodes[prev_timestep].visible = True
                scene_flow_nodes[(prev_timestep - 5) % num_frames].visible = False

            if current_timestep == 0:
                for sn in scene_flow_nodes:
                    sn.visible = False

        prev_timestep = current_timestep
        server.flush()

    # Add top-level frame
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(np.array([0.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )

    frame_nodes = []
    point_static_nodes = []
    point_dynamic_nodes = []
    scene_flow_nodes = []
    point_deform_nodes = []

    st_idx = 0  # Reference frame index for centering

    for i in tqdm(range(st_idx, st_idx + num_frames), desc="Loading frames"):
        frame = video_reader.get_batch([i]).asnumpy()[0]
        scale = 1
        point_map = np.float32(point_maps[i]) * scale
        valid_mask = valid_masks[i]
        deform_mask = deform_masks[i]

        camera_pose = np.float32(camera_poses[i])
        camera_pose[:3, 3] *= scale

        if i < st_idx + num_frames - 1:
            next_camera_pose = np.float32(camera_poses[i + 1])
            next_camera_pose[:3, 3] *= scale
        else:
            next_camera_pose = camera_pose

        # Edge detection for noise removal
        edge_mask = ~point_map_edge(point_map, dist_threshold=0.1)
        valid_mask = valid_mask & edge_mask
        deform_mask = deform_mask & edge_mask

        # Crop boundaries to remove noisy borders
        valid_mask[:10, :] = False
        valid_mask[-40:, :] = False
        valid_mask[:, :20] = False
        valid_mask[:, -20:] = False

        frame = cv2.resize(
            frame,
            (point_map.shape[1], point_map.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        # print("camera pose", camera_pose)

        if scene_flows is not None:
            scene_flow = np.float32(scene_flows[i]) * scale
            new_point_map = point_map + scene_flow

            new_edge_mask = ~point_map_edge(new_point_map, dist_threshold=0.1)
            deform_mask = deform_mask & new_edge_mask

            new_point_xyz = new_point_map.reshape(-1, 3)

        color = frame.reshape(-1, 3)
        point_xyz = point_map.reshape(-1, 3)
        valid_mask = valid_mask.reshape(-1)

        # Convert from camera coordinate to world coordinate
        R = camera_pose[:3, :3]
        T = camera_pose[:3, 3]
        position = (R @ point_xyz.T).T + T

        if i == st_idx:
            center = position[valid_mask].mean(axis=0)

        if scene_flows is not None:
            new_position = (next_camera_pose[:3, :3] @ new_point_xyz.T).T + next_camera_pose[:3, 3]
            motion = np.linalg.norm(new_position - position, axis=1)
            motion_mask = motion > motion.mean() + 3 * motion.std()
            motion_mask = deform_mask.reshape(-1) & motion_mask
            new_position -= center

        position -= center
        camera_pose[:3, 3] -= center

        # Add static and dynamic point clouds
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        point_static_nodes.append(
            server.scene.add_point_cloud(
                name=f"/point_cloud_static/t{i}",
                points=position[valid_mask & ~motion_mask] if scene_flows is not None else position[valid_mask],
                colors=color[valid_mask & ~motion_mask] if scene_flows is not None else color[valid_mask],
                point_size=gui_point_size.value,
                point_shape="rounded",
            )
        )

        point_dynamic_nodes.append(
            server.scene.add_point_cloud(
                name=f"/point_cloud_dynamic/t{i}",
                points=position[valid_mask & motion_mask] if scene_flows is not None else position[valid_mask],
                colors=color[valid_mask & motion_mask] if scene_flows is not None else color[valid_mask],
                point_size=gui_point_size.value,
                point_shape="rounded",
            )
        )

        if scene_flows is not None:
            flow_start = position[valid_mask & motion_mask]
            flow_end = new_position[valid_mask & motion_mask]

            line_points = np.stack([flow_start, flow_end], axis=1)

            num_lines = line_points.shape[0]
            cmap = plt.get_cmap("hsv")
            line_colors = cmap(np.linspace(0, 1, num_lines))[:, :3] * 255
            line_colors = line_colors.astype(np.uint8)
            line_colors = line_colors[:, None, :].repeat(2, axis=1)


            # Randomly sample to reduce visual clutter
            if len(line_points) > 6000:
                indices = np.random.choice(len(line_points), size=6000, replace=False)
                line_points = line_points[indices]
                line_colors = line_colors[indices]

            # Add scene flow line segments
            scene_flow_nodes.append(
                server.scene.add_line_segments(
                    name=f"/scene_flow/t{i}",
                    points=line_points,
                    colors=line_colors,
                    line_width=gui_line_size.value,
                )
            )

            # print(color.max())
            color = (color.astype(np.float32) * 0.6).astype(np.uint8)

            # Add deformed point cloud
            point_deform_nodes.append(
                server.scene.add_point_cloud(
                    name=f"/point_cloud_new/t{i}",
                    points=new_position[valid_mask],
                    colors=color[valid_mask],
                    point_size=gui_point_size.value,
                    point_shape="rounded",
                )
            )


        # Add camera frustum
        fov = -60
        aspect = frame.shape[1] / frame.shape[0]

        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.05,
            image=frame[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(camera_pose[:3, :3]).wxyz,
            position=camera_pose[:3, 3],
        )

        # Add coordinate axes
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )


    # Hide all frames except the active one
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    print("Visualization setup complete. Viewer is running...")

    # Playback Loop
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        # Update point size dynamically
        for pn in point_static_nodes + point_deform_nodes + point_dynamic_nodes:
            pn.point_size = gui_point_size.value

        # Update scene flow line width
        for sf in scene_flow_nodes:
            sf.line_width = gui_line_size.value

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    main()
