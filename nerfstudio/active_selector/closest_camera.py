import numpy as np
import torch
from pathlib import Path
import os
import yaml

from nerfstudio.cameras.cameras import Cameras

from nerfstudio.active_selector.idx_to_cam import idx_to_cam
import modern_robotics as mr

def poses_from_cams(cams) -> np.ndarray:
    Rp = cams
    arr_SE3 = np.hstack((Rp, np.zeros((len(Rp), 1, 4))))
    arr_SE3[:, -1, -1] = 1.0
    return arr_SE3

def find_closest_cam(pose, cams_poses, num=1):
    cand_poses = poses_from_cams(cams_poses)
    lie_dist = np.zeros(len(cand_poses))
    for idx, cp in enumerate(cand_poses):
        lie_dist[idx] = quadratic_cost_SE3(pose, cp)
    
    selected = np.zeros((num, 4, 4))
    selected_indices = []
    for idx in range(num):
        selected_idx = lie_dist.argmin()
        selected_indices.append(selected_idx)
        selected[idx] = (cand_poses[selected_idx])
        lie_dist = np.delete(lie_dist, selected_idx)
        cand_poses = np.delete(cand_poses, selected_idx)
    
    return selected, selected_indices

def quadratic_cost_SE3(g1, g2, M = np.eye(6)) -> float:
    '''
    quadatic cost equation (eq 29)
    adapted from https://murpheylab.github.io/pdfs/2016RSSFaMu.pdf
    params:
    - g1: SE(3), 4x4 matrix
    - g2: SE(3), 4x4 matrix
    - M: 4x4 matrix

    returns:
    float value of cost
    '''
    diff_mat = mr.TransInv(g2) @ g1
    diff_vec = np.expand_dims(mr.se3ToVec(mr.MatrixLog6(diff_mat)), axis=1)
    return (diff_vec.T @ M @ diff_vec).flatten()[0]

if __name__ == "__main__":
    os.chdir("/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs")
    print(os.getcwd())
    method = "bnn"
    trial = 0
    load_config = "outputs/BM_synthetic_lego/nerfacto/" + method + "_test" + str(trial) + "/config.yml"

    path = Path(load_config)
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    config.vis = "None"

    if "nerfstudio_models" in os.listdir(path.parent.absolute()):
        checkpoint_name = sorted(os.listdir(path.parent.absolute().__str__() + "/nerfstudio_models"))[-1]
        config.load_checkpoint = Path(path.parent.absolute().__str__() + "/nerfstudio_models/" + checkpoint_name)

    # config.iterative_training = True
    from nerfstudio.engine.trainer import Trainer
    trainer = Trainer(config=config)
    trainer.setup(test_mode="test")
    pipeline = trainer.pipeline
    pipeline.eval()

    num_predictions = 10
    pipeline.model.num_preds = num_predictions
    pipeline.model.output_preds = False

    current_cams = idx_to_cam(load_config[:-10] + "candidates.txt", trainer.pipeline.config.datamanager.data, split="train", scale_factor=0.1)
    pose = (current_cams.camera_to_worlds[0]).numpy()
    pose = np.vstack((pose, np.zeros((1, 4))))
    pose[-1, -1] = 1.0
    print(pose)
    find_closest_cam(pose, current_cams)