# NeRF implementation of kernel ergodic search, over hemisphere

import torch
import numpy as np
from tqdm import tqdm
from typing import List
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
import yaml
from dataclasses import dataclass

from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.utils.io import load_from_json

import argparse
from nerfstudio.active_selector.kes_so3.hemisphere import spherical_to_hemi
from nerfstudio.active_selector.kes_so3.lie_agent import Agent
from nerfstudio.active_selector.kes_so3.so3_cost import quadratic_cost
import subprocess as sp

@dataclass
class CropData:
    """Data for cropping an image."""

    background_color: torch.Tensor = torch.Tensor([0.0, 0.0, 0.0])
    """background color"""
    obb: OrientedBox = OrientedBox(R=torch.eye(3), T=torch.zeros(3), S=torch.ones(3) * 2)
    """Oriented box representing the crop region"""

    # properties for backwards-compatibility interface
    @property
    def center(self):
        return self.obb.T

    @property
    def scale(self):
        return self.obb.S

def get_crop_from_json(camera_json) -> CropData:
    """Load crop data from a camera path JSON

    args:
        camera_json: camera path data
    returns:
        Crop data
    """
    if "crop" not in camera_json or camera_json["crop"] is None:
        return None
    bg_color = camera_json["crop"]["crop_bg_color"]
    center = camera_json["crop"]["crop_center"]
    scale = camera_json["crop"]["crop_scale"]
    rot = (0.0, 0.0, 0.0) if "crop_rot" not in camera_json["crop"] else tuple(camera_json["crop"]["crop_rot"])
    assert len(center) == 3
    assert len(scale) == 3
    assert len(rot) == 3
    return CropData(
        background_color=torch.Tensor([bg_color["r"] / 255.0, bg_color["g"] / 255.0, bg_color["b"] / 255.0]),
        obb=OrientedBox.from_params(center, rot, scale),
    )

class ActiveSelector():

    def __init__(self, save_data: bool = True, num_preds: int = 10, json_file = None) -> None:
        self.num_preds = num_preds
        self.save_data = save_data
        if json_file is None:
            json_file = "/home/ayush/Desktop/datasets/nerf_synthetic/lego/camera_paths/synthetic_lego.json"
        with open(json_file, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        self.crop_data: CropData = get_crop_from_json(camera_path)
        self.background, self.obb = self.crop_data.background_color, self.crop_data.obb
        self.agent = Agent()
        self.radius = self.obb.S[0] * 1.5
        self.height = self.obb.S[0] * 1.5
        self.ground_position = self.obb.T.numpy() - np.array([0.0, 0.0, self.obb.S[-1]/2]) # np.array([0.0, 0.0, -.5 - 0.35])
        self.center_of_mass = self.obb.T.detach().numpy() #np.array([0, 0, -.5])  # Example variable point above the origin
        top_view = np.copy(self.center_of_mass)
        top_view[2] = self.height
        self.scale_r = np.linalg.norm(self.center_of_mass - top_view)
        self.num_samples = 500
        if self.num_samples > 500:
            input("must re-render if more than 500 random poses are desired")

    def hemi_points(self, pipeline):
        rendered_ang = "data/transforms_300.json"
        meta = load_from_json(Path(rendered_ang))
        poses = []
        hemi_angles = []
        for i, frame in enumerate(meta["frames"]):
            if i >= self.num_samples:
                break
            poses.append(np.array(frame["transform_matrix"]))
            hemi_angles.append(np.array(frame["hemi_angles"]))
        poses = np.array(poses).astype(np.float32)
        self.rotation_matrices = poses[:, :-1, :-1]
        # hemi_angles = uniform_hemi_angles(num_samples=self.num_samples, longitude_range=[0., 2*np.pi], latitude_range=[0., np.pi/2], seed=int(str.split(rendered_ang, ".json")[0][-1]))
        # self.hemi_angles = hemi_angles.copy()
        self.hemi_angles = np.array(hemi_angles).astype(np.float32)
        poses = torch.from_numpy(poses[:, :3])
        poses[..., 3] *= pipeline.datamanager.dataparser.scale_factor
        return poses
    
    def uncertainty_by_pose_calc(self, pipeline, c2ws = None):
        # returns Nx7 array
        # [translation_x, translation_y, translation_z, roll, pitch, yaw, uncertainty value]
        # background = torch.zeros(3, device=pipeline.device)
        # pipeline.model.renderer_rgb.background_color = background
        pipeline.eval()

        if c2ws is None:
            c2ws = self.hemi_points(pipeline)
        # resolution = np.array([400, 400])
        resolution = np.array([800, 800])
        fov = 75.0
        fx = three_js_perspective_camera_focal_length(fov, resolution[0])
        fy = fx
        cx = resolution[1] / 2
        cy = resolution[0] / 2
        mask_val = 0.
        unc_value = np.zeros(len(c2ws))

        test_cam = Cameras(fx=torch.tensor(fx), 
                        fy=torch.tensor(fy), 
                        cx=torch.tensor(cx),
                        cy=torch.tensor(cy), 
                        camera_type=torch.tensor([1]),
                        camera_to_worlds=torch.zeros((3, 4)))
        
        for i, pose in tqdm(enumerate(c2ws), desc="Calculating uncertainty values", total=len(c2ws)):
            pipeline.eval()
            # r = np.linalg.norm(pose[:, -1].flatten() - self.center_of_mass)**2
            r = np.linalg.norm(pose[:, -1].flatten())**2
            middle = 200
            test_cam.camera_to_worlds = pose
            with background_color_override_context(torch.tensor([0.0, 0.0, 0.0], device=pipeline.device)), torch.no_grad():
                outs = pipeline._model.get_outputs_for_camera(test_cam, obb_box=self.obb, middle=middle)
            # scale = (torch.max(outs["entropy"]) - torch.min(outs["entropy"]))
            # outs["entropy"] = (outs["entropy"] - torch.min(outs["entropy"]))/scale
            # outs["entropy"] = torch.sum(outs["entropy"], axis=-1)/3
            # mask = (outs["entropy"] < mask_val).clone()
            # outs["entropy"].masked_fill_(mask, 0.0)
            # outs["entropy"] *= scale
            # unc_value[i] = np.sum(outs["entropy"].numpy())/np.prod((outs['entropy'].numpy().shape)) # * (.75*np.tanh((1/self.radius*(r - self.scale_r) - 1)) + 1)
            unc_value[i] = np.mean(outs["entropy"].detach().numpy())
            # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
            # ax[0].set_title(f"Cropped RGB at {np.array2string(self.hemi_angles[i].flatten(), precision=2, floatmode='fixed')}", fontsize=10)
            # ax[1].set_title(f"Entropy value {(unc_value[i]):.2f}", fontsize=10)
            # # ax[1].set_title(f"Entropy value {(unc_value[i] - rgb_std_dev_val):.2f}", fontsize=10)
            # ax[0].imshow(outs['rgb'])
            # ax[1].imshow(outs['entropy'])
            # plt.show()
            # plt.close()

        pose_unc_correlation = np.zeros((len(unc_value), 3+1))
        for count, (pose, val) in enumerate(zip(self.hemi_angles, unc_value)):
            pose_unc_correlation[count] = np.concatenate([pose, [val]])
        return pose_unc_correlation
    
    def view_pose_unc_correlation(self, pose_unc_corr, unc):
        hemi_cart_points = self.hemi_points(pipeline)[:, -1].detach().cpu().numpy()
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        sc = ax.scatter(hemi_cart_points[:, 0], 
                        hemi_cart_points[:, 1], 
                        hemi_cart_points[:, 2], 
                        c=unc, cmap='magma', label="unc", alpha=0.5,
                        s=50, linewidth=0)
                        # c=pose_unc_corr[:, -1], cmap='magma', label="unc", alpha=0.5)
        cbar = plt.colorbar(sc)
        cbar.set_label('Values')
        ax.scatter(0., 0., 0., c='red', s=50, alpha=0.75)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-.4, .4])
        ax.set_ylim([-.4, .4])
        ax.set_zlim([0., .4])
        plt.title('Original Samples Uncertainty Colormap')
        plt.show()

    def resampling_alg(self, pose_unc_corr, resampling_criteria=5, num_views=10, dist=None): # vals=[1.25, 1.23, 1.22]):
        # prediction
        xbar = pose_unc_corr[:, :-1]
        unc = pose_unc_corr[:, -1]
        # get weights and normalize
        weights = unc / np.sum(unc)
        weights = unc**3
        weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        print(np.min(weights), np.max(weights), np.mean(weights))
        
        if dist is not None:
            print(np.min(dist), np.max(dist), np.mean(dist))
            weights += dist
            weights /= np.max(weights)
            np.savetxt("points_erg.txt", np.hstack((xbar, weights[:, None])))
            input("saved")

        # resampling algorithm
        count = 0
        
        print(np.min(weights), np.max(weights), np.mean(weights))
        all_samp = np.zeros((num_views, 3))
        counts = np.zeros(num_views)
        count = 0
        while count < num_views:
            print("relooping")
            for index, w in tqdm(enumerate(weights)):
                nu = np.random.uniform(low=0.05, high=1)
                if w > resampling_criteria * nu:
                    alr_in = False
                    for s in all_samp:
                        # if np.all(s == xbar[index, :]): # or np.abs(s[2] - xbar[index, 2]) < 0.1:
                        # if np.all(s == xbar[index, :]) or xbar[index, 0] > 0.75:
                        if np.all(s == xbar[index, :]) or xbar[index, 0] < 0.4:
                            alr_in = True
                            print("alr in")
                            break
                    if not alr_in:
                        all_samp[count] = xbar[index, :]
                        print(count, all_samp[count], w)
                        counts[count] = index
                        count += 1
                        if count >= num_views:
                            break
        return all_samp, counts.astype(np.int64), weights

    def nbv_resampled(self, pipeline, num_views, curr_so3=None):
        # pose_unc = self.uncertainty_by_pose_calc(pipeline)
        # np.savetxt("points.txt", pose_unc)
        pose_unc = np.loadtxt("points.txt")
        unc = pose_unc[:, -1]
        # weights = unc**3
        # weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
        # weights += curr_so3
        # self.view_pose_unc_correlation(self.hemi_points(pipeline), weights)
        # self.view_pose_unc_correlation(self.hemi_points(pipeline), unc)
        # self.view_pose_unc_correlation(self.hemi_points(pipeline), curr_so3)
        # self.view_pose_unc_correlation(self.hemi_points(pipeline), pose_unc[:, -1])
        # plt.plot(pose_unc[:, -1], 'bo')
        # plt.show()
        # plt.close()
        
        resampled_rpy, counts, weights = self.resampling_alg(pose_unc, resampling_criteria=15, num_views=num_views, dist=curr_so3)
        
        return resampled_rpy

    def rpy_to_SE3(self, best_rpy):
        rotation = self.agent.eulerToSo3(best_rpy)
        # print((rotation, np.expand_dims(best_xyzrpy[:3], axis=1)))
        c2ws = np.hstack((rotation, np.expand_dims(spherical_to_hemi([best_rpy[1], best_rpy[0]], self.radius, self.height, origin=self.ground_position), axis=1)))
        cam_SE3 = np.vstack((c2ws, np.zeros((1, 4))))
        cam_SE3[-1, -1] = 1.0
        return cam_SE3
    
    def dist_to_others(self, cand_cams, curr_cams):
        dist_sum = np.zeros(len(cand_cams))
        M = np.eye(3)
        # M[2, 2] *= .4
        for i, c in (enumerate(cand_cams)):
            for curr in curr_cams:
                # dist_sum[i] += np.linalg.norm(c - curr)
                dist_sum[i] += quadratic_cost(c, curr, M)
        return dist_sum


def run(trainer, load_config_dir, num_views=1, json_file=None, nbv=False):
    # os.chdir("/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs")
    os.makedirs(load_config_dir, exist_ok=True)
    os.makedirs(load_config_dir + "/data", exist_ok=True)
    os.makedirs(load_config_dir + "/data/images", exist_ok=True)

    #TODO: change to your blender path
    blender_path = "/home/ayush/Downloads/blender-4.1.1-linux-x64/blender"
    object_file = "lego.blend"
    command = "cd " + os.getcwd() + "/data/blend_files && " + blender_path + " -b " + object_file + " --python 360_view.py {0} {1} -- --cycles-device CUDA".format(os.path.abspath(load_config_dir) + "/data/images", load_config_dir + "/data/transforms_train.json")

    curr_cams = np.loadtxt(str(trainer.config.get_base_dir()) + "/data/selected.txt").astype(np.int64)
    data_json = str(trainer.config.pipeline.datamanager.dataparser.data) +  "/transforms_train.json"
    meta = load_from_json(Path(data_json))
    so3_cams = np.zeros((len(curr_cams), 3, 3))
    hemi_angles = np.zeros((len(curr_cams), 3))
    for i, frame in enumerate(np.asarray(meta["frames"])[curr_cams]):
        so3_cams[i] = (np.array(frame["transform_matrix"])[:-1, :-1])
        hemi_angles[i] = (np.array(frame["hemi_angles"]))

    erg = ActiveSelector(save_data=True, json_file=json_file)
    if so3_cams is not None:
        agent = Agent()
        erg.hemi_points(trainer.pipeline)
        # dist = erg.dist_to_others(erg.rotation_matrices, so3_cams)
        erg.hemi_angles[:, 0] = 0
        dist = erg.dist_to_others(agent.eulerToSo3(erg.hemi_angles), agent.eulerToSo3(hemi_angles))
        # dist = erg.dist_to_others(erg.hemi_angles, hemi_angles)
        dist -= np.min(dist)
        dist /= np.max(dist)
    if not nbv:
        resampled_rpy = erg.nbv_resampled(trainer.pipeline, num_views=num_views, curr_so3=None)
    else:
        if num_views == 1:
            ## max_dist --> adding 6 views, but using this as a boolean check to run 3 diff experiments
            weights = dist.copy()
            resampled_rpy = erg.hemi_angles[np.flip(np.argsort(weights))[:6]]
        else:
            pose_unc = erg.uncertainty_by_pose_calc(trainer.pipeline)
            np.savetxt("points.txt", pose_unc)
            pose_unc = np.loadtxt("points.txt")
            unc = pose_unc[:, -1]
            mask = pose_unc[:, 0] > 0.75
            unc[mask] = 0.
            weights = unc**3
            weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
            weights += dist
            resampled_rpy = pose_unc[np.flip(np.argsort(weights))[:num_views], :-1]

    np.savetxt(load_config_dir + "/data/rpy_traj.txt", resampled_rpy)

    start_index = 116 - trainer.config.subset_data + len(curr_cams)
    adding_index = np.arange(start_index, start_index+num_views)
    new_fp = os.path.abspath(load_config_dir) + "/data/images/transforms.json"
    sp.call(command, shell=True)
    print("generated new images (:")

    trainer.pipeline.model.output_preds = False
    # return np.concatenate((curr_cams, np.asarray(np.loadtxt(load_config_dir + "/data/candidates.txt").astype(np.int64)[adding_index]).flatten()))
    original_json = str(trainer.config.pipeline.datamanager.dataparser.data) +  "/transforms_train.json"
    combine_json_files([original_json, new_fp], os.path.abspath(load_config_dir) + "/data/transforms_train.json")
    original_json = str(trainer.config.pipeline.datamanager.dataparser.data) +  "/transforms_val.json"
    combine_json_files([original_json], os.path.abspath(load_config_dir) + "/data/transforms_val.json")
    original_json = str(trainer.config.pipeline.datamanager.dataparser.data) +  "/transforms_test.json"
    combine_json_files([original_json], os.path.abspath(load_config_dir) + "/data/transforms_test.json")
    return np.concatenate((curr_cams, adding_index))

def combine_json_files(file_paths, output_file):
    if not file_paths:
        print("No files to merge.")
        return

    # Read the first file to initialize the combined_data
    with open(file_paths[0], 'r') as file:
        combined_data = json.load(file)
        for frame in combined_data["frames"]:
            if frame["file_path"].__contains__("./"):
                frame["file_path"] = str(str.split(file_paths[0], "transforms")[0] / Path(frame["file_path"].replace("./", "")))

    # Iterate over the remaining files and merge the "frames" lists
    for file_path in file_paths[1:]:
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skipping.")
            continue

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                if isinstance(data, dict) and 'frames' in data:
                    combined_data['frames'].extend(data['frames'])
                else:
                    print(f"File {file_path} does not have the expected structure. Skipping.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file {file_path}: {e}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    # Write the combined data to the output file
    try:
        with open(output_file, 'w') as output_file:
            json.dump(combined_data, output_file, indent=4)
        print(f"Combined JSON written to {output_file}")
    except Exception as e:
        print(f"Error writing to output file {output_file}: {e}")

if __name__ == "__main__":
    # python nerfstudio/active_selector/fisher_rf.py --load-config /home/ayush/Desktop/nerf/uncertainty_NeRFs/outputs/512_rays/nerfacto/2024-05-07_172703
    parser = argparse.ArgumentParser(description='Args necessary for active selector')
    parser.add_argument('--load-config-dir', type=str, help='dir with config file for trainer')
    parser.add_argument('--dir-to-save', type=str, help='dir where images will be saved')
    
    args = parser.parse_args()
    os.chdir("/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs")
    path = Path(args.load_config_dir)
    
    config = yaml.load(path.read_text(), Loader=yaml.Loader)
    config.vis = "None"
    if "nerfstudio_models" in os.listdir(path.parent.absolute()):
        checkpoint_name = sorted(os.listdir(path.parent.absolute().__str__() + "/nerfstudio_models"))[0]
        config.load_checkpoint = Path(path.parent.absolute().__str__() + "/nerfstudio_models/" + checkpoint_name)

    from nerfstudio.engine.trainer import Trainer
    # config.iterative_training = True
    config.subset_data = config.max_data
    trainer = Trainer(config=config)
    trainer.setup(test_mode="val")
    pipeline = trainer.pipeline
    trainer.pipeline.datamanager.test_split = "val"
    background = torch.zeros(3, device=pipeline.device)
    pipeline.model.renderer_rgb.background_color = background
    pipeline.eval()
    print(run(trainer, args.dir_to_save, json_file = "/data/bounding_boxes/chair_synthetic.json", num_views=6))
