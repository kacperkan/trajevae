from typing import Any, Dict, List, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from trajevae.common import copy2cpu
from trajevae.utils.skeleton import Skeleton


def visualize_poses(
    batch_poses: torch.Tensor,
    every_nth_pose: int,
    skeleton: Skeleton,
    meta_data: Dict[str, Any],
    trajectories: torch.Tensor,
    canvas_width: int = 256,
    canvas_height: int = 256,
) -> torch.Tensor:
    plt.switch_backend("agg")
    images = []
    size = 6
    azim = 120

    batch_poses_np = copy2cpu(batch_poses)
    trajectories_np = copy2cpu(trajectories)
    was_not_trajectory_used = np.sum(trajectories_np, axis=(1, 3)) == 0

    for batch_index in range(len(batch_poses_np)):
        poses = batch_poses_np[batch_index]
        fig = plt.figure(figsize=(size, size))
        canvas = FigureCanvas(fig)
        radius = 1.7
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.view_init(elev=15.0, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect("equal")
        except NotImplementedError:
            ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        character_bound = poses[:, 0, [0, 1]]
        ax.set_xlim3d(
            [
                -radius / 2 + character_bound[:, 0].min(),
                radius / 2 + character_bound[:, 0].max(),
            ]
        )
        ax.set_ylim3d(
            [
                -radius / 2 + character_bound[:, 1].min(),
                radius / 2 + character_bound[:, 1].max(),
            ]
        )

        parents = skeleton.parents()
        alphas = (
            np.arange(1, poses.shape[0] + 1, dtype=np.float32) / poses.shape[0]
        )

        for i in range(0, poses.shape[0], every_nth_pose):
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = "red" if j in skeleton.joints_right() else "black"
                pos = poses[i]
                ax.plot(
                    [pos[j, 0], pos[j_parent, 0]],
                    [pos[j, 1], pos[j_parent, 1]],
                    [pos[j, 2], pos[j_parent, 2]],
                    zdir="z",
                    c=col,
                    alpha=alphas[i],
                )
        for traj_index in range(trajectories_np.shape[-2]):
            if was_not_trajectory_used[batch_index, traj_index]:
                continue
            xs = trajectories_np[batch_index, :, traj_index, 0]
            ys = trajectories_np[batch_index, :, traj_index, 1]
            zs = trajectories_np[batch_index, :, traj_index, 2]
            ax.scatter(xs, ys, zs, s=20, color="blue")

        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
            canvas.get_width_height()[::-1] + (3,)
        )
        image = cv2.resize(image, (canvas_width, canvas_height))

        images.append(
            torch.from_numpy(image).float().div(255).permute((2, 0, 1))
        )
        plt.close()
    return torch.stack(images, dim=0)


def get_video_poses(
    initial_pose: torch.Tensor,
    batch_poses: torch.Tensor,
    skeleton: Skeleton,
    trajectories: torch.Tensor,
    canvas_width: int = 256,
    canvas_height: int = 256,
    visualize_separate: bool = False,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[List[np.ndarray]]]]:
    plt.switch_backend("agg")

    size = 6
    azim = 120
    radius = 1.7
    elev = 15.0

    def _create_single_frame(
        current_pose: np.ndarray,
    ) -> Tuple[FigureCanvas, plt.Axes]:
        fig = plt.figure(figsize=(size, size))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([-radius / 2, radius / 2])
        try:
            ax.set_aspect("equal")
        except NotImplementedError:
            ax.set_aspect("auto")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 7.5
        character_bound = current_pose[:, 0, [0, 1]]
        ax.set_xlim3d(
            [
                -radius / 2 + character_bound[:, 0].min(),
                radius / 2 + character_bound[:, 0].max(),
            ]
        )
        ax.set_ylim3d(
            [
                -radius / 2 + character_bound[:, 1].min(),
                radius / 2 + character_bound[:, 1].max(),
            ]
        )

        return canvas, ax

    images = []

    batch_poses_np = copy2cpu(batch_poses)
    trajectories_np = copy2cpu(trajectories)
    initial_poses_np = copy2cpu(initial_pose)
    parents = skeleton.parents()

    traj_images = []
    for batch_index in range(len(batch_poses_np)):
        poses = batch_poses_np[batch_index]
        init_pose = initial_poses_np[batch_index]

        if visualize_separate:
            for traj_index in range(trajectories_np.shape[-2]):
                c, a = _create_single_frame(poses)
                xs = trajectories_np[batch_index, :, traj_index, 0]
                ys = trajectories_np[batch_index, :, traj_index, 1]
                zs = trajectories_np[batch_index, :, traj_index, 2]
                a.scatter(xs, ys, zs, s=20, color="blue")

                c.draw()
                image = np.frombuffer(
                    c.tostring_rgb(), dtype=np.uint8
                ).reshape(c.get_width_height()[::-1] + (3,))
                image = cv2.resize(image, (canvas_width, canvas_height))

                traj_images.append(image)
                plt.close()

        sample_sequence = []

        for i in range(0, poses.shape[0]):
            canvas, ax = _create_single_frame(poses)

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = "red" if j in skeleton.joints_right() else "black"
                col_init = (
                    "purple" if j in skeleton.joints_right() else "green"
                )
                pos = poses[i]
                ax.plot(
                    [init_pose[j, 0], init_pose[j_parent, 0]],
                    [init_pose[j, 1], init_pose[j_parent, 1]],
                    [init_pose[j, 2], init_pose[j_parent, 2]],
                    zdir="z",
                    c=col_init,
                )
                ax.plot(
                    [pos[j, 0], pos[j_parent, 0]],
                    [pos[j, 1], pos[j_parent, 1]],
                    [pos[j, 2], pos[j_parent, 2]],
                    zdir="z",
                    c=col,
                )
            for traj_index in range(trajectories_np.shape[-2]):
                xs = trajectories_np[batch_index, :, traj_index, 0]
                ys = trajectories_np[batch_index, :, traj_index, 1]
                zs = trajectories_np[batch_index, :, traj_index, 2]
                ax.scatter(xs, ys, zs, s=20, color="blue")

            canvas.draw()
            image = np.frombuffer(
                canvas.tostring_rgb(), dtype=np.uint8
            ).reshape(canvas.get_width_height()[::-1] + (3,))
            image = cv2.resize(image, (canvas_width, canvas_height))

            sample_sequence.append(image)
            plt.close()
        images.append(np.stack(sample_sequence, axis=0))
    if visualize_separate:
        return images, traj_images
    return images
