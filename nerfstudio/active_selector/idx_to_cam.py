import numpy as np
import torch
from pathlib import Path
import imageio

from nerfstudio.utils.io import load_from_json
from nerfstudio.cameras.cameras import Cameras, CameraType


def idx_to_cam(txt, data_dir, split="train", scale_factor=1.0) -> Cameras:
    inds = np.loadtxt(txt)
    meta = load_from_json(data_dir / f"transforms_{split}.json")
    image_filenames = []
    poses = []

    if len(inds.shape) == 0:
        inds = np.array([inds], dtype=int)
    frames = [meta["frames"][int(ind)] for ind in inds]
    for frame in frames:
        fname = data_dir / Path(frame["file_path"].replace("./", "") + ".png")
        image_filenames.append(fname)
        poses.append(np.array(frame["transform_matrix"]))
    poses = np.array(poses).astype(np.float32)
    poses[:, :3, -1] *= 0.1

    img_0 = imageio.v2.imread(image_filenames[0])
    # center_crop = 650
    # img_0 = img_0[(img_0.shape[0]//2-center_crop//2):(img_0.shape[0]//2+center_crop//2),
    #                 (img_0.shape[1]//2-center_crop//2):(img_0.shape[1]//2+center_crop//2),
    #                 :]
    image_height, image_width = img_0.shape[:2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

    cx = image_width / 2.0
    cy = image_height / 2.0
    camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

    # in x,y,z order
    camera_to_world[..., 3] *= scale_factor

    return Cameras(
        camera_to_worlds=camera_to_world,
        fx=focal_length,
        fy=focal_length,
        cx=cx,
        cy=cy,
        camera_type=CameraType.PERSPECTIVE,
    )