# NeRF implementation of kernel ergodic search, over hemisphere

import torch
import numpy as np
from tqdm import tqdm
from typing import List
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.spatial.transform import Rotation as R
import yaml
from dataclasses import dataclass

from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.cameras.cameras import Cameras, CameraType, RayBundle
from nerfstudio.utils.io import load_from_json

from sklearn.mixture import GaussianMixture

import argparse
from nerfstudio.active_selector.idx_to_cam import idx_to_cam
from nerfstudio.active_selector.closest_camera import find_closest_cam
from nerfstudio.active_selector.kes_so3.hemisphere import uniform_hemi_angles, vector_to_quaternion, generate_hemi_angles, spherical_to_hemi, calculate_normal_vectors, rpy_to_spherical
from nerfstudio.active_selector.kes_so3.ilqr_kes import Agent, Distr, ergodic_search
from nerfstudio.active_selector.kes_so3.so3_cost import quadratic_cost
from nerfstudio.active_selector.kes_so3.optimize_params import KernelObjective
import subprocess as sp
import time

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

class ErgodicSelector():

    def __init__(self, dir_to_save: str, save_data: bool = True, num_preds: int = 10, json_file = None) -> None:
        self.num_preds = num_preds
        self.save_data = save_data
        self.dir_to_save = dir_to_save
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
        rendered_ang = os.pardir + "data/transforms_300.json"
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
    
    def generate_c2ws(self, hemi_angles, hemi_cart_points):
        normals = calculate_normal_vectors(hemi_cart_points, self.center_of_mass)
        rotation_matrices = np.zeros((len(normals), 3, 3))
        for i, (angle, n_vec) in enumerate(zip(hemi_angles, normals)):
            current_orient = vector_to_quaternion(n_vec).as_matrix()
            target_orient = (R.from_matrix(np.eye(3)) * R.from_euler('z', angle[0] - np.pi))
            dot = np.dot(current_orient[:, 0], target_orient.as_matrix()[:, 0]) / np.linalg.norm(current_orient[:, 0]) / np.linalg.norm(target_orient.as_matrix()[:, 0])
            if dot > 1:
                dot = 1.0
            rotate = np.arccos(dot)
            rotation_matrices[i] = (target_orient * R.from_euler('y', rotate) * R.from_euler('y', -np.pi/2) * R.from_euler('z', -np.pi/2)).as_matrix()
        # rotation, _ = angle_from_normal(normals, points_per_circle=points_per_circle)
        c2ws = torch.from_numpy(np.concatenate((rotation_matrices, np.expand_dims(hemi_cart_points, axis=2)), axis=2))
        return c2ws
    
    def uncertainty_by_pose_calc(self, pipeline, c2ws = None):
        # returns Nx7 array
        # [translation_x, translation_y, translation_z, roll, pitch, yaw, uncertainty value]
        # background = torch.zeros(3, device=pipeline.device)
        # pipeline.model.renderer_rgb.background_color = background
        pipeline.eval()
        # num_predictions = 3
        # pipeline.model.num_preds = num_predictions
        # pipeline.model.output_preds = True

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
                # outs = pipeline._model.get_outputs_for_camera(test_cam, obb_box=None, middle=middle)
            # outs["rgb_std_dev"] = (outs["rgb_std_dev"] - torch.min(outs["rgb_std_dev"]))/(torch.max(outs["rgb_std_dev"]) - torch.min(outs["rgb_std_dev"]))
            # # outs['rgb_std_dev'].masked_fill_((torch.logical_and(outs["rgb_std_dev"] < 1e-6, outs['rgb'] > 1e-6)), 1.0)
            # # outs['rgb_std_dev'].masked_fill_((torch.logical_and(outs["rgb_std_dev"] < 1e-5, outs['rgb'] > .99)), 1.0)
            # # outs['rgb_std_dev'].masked_fill_((outs["rgb_std_dev"] < 1e-5), 1.0)
            # outs['rgb_std_dev'] = torch.nan_to_num(outs['rgb_std_dev'])
            # outs['rgb_std_dev'] = torch.sum(torch.abs(outs['rgb_std_dev']), axis=-1)/3
            # mask = (outs['rgb_std_dev'] < mask_val).clone()
            # outs['rgb_std_dev'].masked_fill_(mask, 0.0)
            # scale = (torch.max(outs["entropy"]) - torch.min(outs["entropy"]))
            # outs["entropy"] = (outs["entropy"] - torch.min(outs["entropy"]))/scale
            # outs["entropy"] = torch.sum(outs["entropy"], axis=-1)/3
            # mask = (outs["entropy"] < mask_val).clone()
            # outs["entropy"].masked_fill_(mask, 0.0)
            # outs["entropy"] *= scale
            # unc_value[i] = np.sum(outs["entropy"].numpy())/np.prod((outs['entropy'].numpy().shape)) # * (.75*np.tanh((1/self.radius*(r - self.scale_r) - 1)) + 1)
            unc_value[i] = np.mean(outs["entropy"].detach().numpy())
            # percent_in_image = (np.count_nonzero(outs["rgb"].numpy()) / np.prod((outs['rgb'].numpy().shape)))
            # unc_value[i] = np.sum(outs["rgb_std_dev"].numpy())/np.prod((outs['rgb_std_dev'].numpy().shape)) * (-.75*np.tanh((2.5*percent_in_image - 1)) + 1)
            # rgb_std_dev_val = np.sum(outs["rgb_std_dev"].numpy())/np.prod((outs['rgb_std_dev'].numpy().shape)) # * (.75*np.tanh((1/self.radius*(r - self.scale_r) - 1)) + 1)
            # print(unc_value[i], rgb_std_dev_val)
            # unc_value[i] += rgb_std_dev_val
            # fig, ax = plt.subplots(1, 2, figsize=(10, 8))
            # # fig, ax = plt.subplots(1, 3, figsize=(12, 8))
            # ax[0].set_title(f"Cropped RGB at {np.array2string(self.hemi_angles[i].flatten(), precision=2, floatmode='fixed')}", fontsize=10)
            # ax[1].set_title(f"Entropy value {(unc_value[i]):.2f}", fontsize=10)
            # # ax[1].set_title(f"Entropy value {(unc_value[i] - rgb_std_dev_val):.2f}", fontsize=10)
            # # ax[2].set_title(f"RGB unc value {rgb_std_dev_val:.2f}", fontsize=10)
            # # ax[2].imshow(outs['rgb_std_dev'])
            # ax[0].imshow(outs['rgb'])
            # ax[1].imshow(outs['entropy'])
            # plt.show()
            # plt.close()

        pose_unc_correlation = np.zeros((len(unc_value), 3+1))
        for count, (pose, val) in enumerate(zip(self.hemi_angles, unc_value)):
            pose_unc_correlation[count] = np.concatenate([pose, [val]])
        return pose_unc_correlation
    
    def view_pose_unc_correlation(self, pose_unc_corr, unc):
        hemi_cart_points = pose_unc_corr[..., 3]
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
        xhat = xbar.copy() # np.zeros((150, 6))
        new_unc = unc.copy()
        hi = np.min([1.0, np.max(weights)])
        # resampling_criteria = np.sort(weights)[len(weights)//6*5] / np.max(weights)
        # print(np.sort(weights))
        # print(np.sort(weights)[-10:], xbar[np.argsort(weights)[-10:]])
        # resampling_criteria = vals
        # xhat = np.tile(xbar[np.argsort(weights)[-10:]], (10, 1))
        # counts = np.tile(np.argsort(weights)[-10:], 10)
        # return xhat, counts

        print(np.min(weights), np.max(weights), np.mean(weights))
        all_samp = np.zeros((num_views, 3))
        counts = np.zeros(num_views)
        count = 0
        while count < num_views:
            print("relooping")
            for index, w in tqdm(enumerate(weights)):
                nu = np.random.uniform(low=0.05, high=1)
                if w > resampling_criteria * nu:
                    # xhat[count, :] = xbar[index, :]
                    alr_in = False
                    for s in all_samp:
                        # if np.all(s == xbar[index, :]) or xbar[index, 0] > 1.25: # or np.abs(s[2] - xbar[index, 2]) < 0.1:
                        if np.all(s == xbar[index, :]) or xbar[index, 0] > 0.75: # or np.abs(s[2] - xbar[index, 2]) < 0.1:
                        # if np.all(s == xbar[index, :]) or xbar[index, 0] < 0.4:
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

        # weights = unc**3 / np.max(unc**3)
        # all_samp = []
        # counts = np.zeros(len(weights))
        # while count < len(xhat):
        #     for index, w in enumerate(weights):
        #         nu = np.random.uniform(low=0.25, high=1)
        #         if w > resampling_criteria * nu:
        #             # xhat[count, :] = xbar[index, :]
        #             all_samp.append(xbar[index, :])
        #             counts[count] = index
        #             count += 1
        #             if count >= len(xhat):
        #                 break
        # print("selected / total:", len(np.unique(xhat, axis=0)), "/", len(np.unique(xbar, axis=0)))

        # start_time = time.time()
        # counts = np.zeros(len(weights))
        # while count < len(xhat):
        #     for index, w in enumerate(weights):
        #         nu = np.random.uniform(low=np.min(weights), high=hi)
        #         if w > resampling_criteria * nu:
        #             xhat[count, :] = xbar[index, :]
        #             new_unc[count] = unc[index]
        #             counts[count] = index
        #             count += 1
        #             if count >= len(xhat):
        #                 break
        #     if time.time() - start_time > 1:
        #         xhat = xbar.copy()
        #         vals -= 0.05
        #         break
        
        # if len(np.unique(xhat, axis=0)) > 10:
        #     xhat, counts = self.resampling_alg(pose_unc_corr, vals=vals+0.1)
        # else:
        # fig = plt.figure(figsize=(8, 6))
        # ax = fig.add_subplot(111)
        # for ang, alph in zip(xhat, weights / np.max(weights)):
        #     ax.scatter(ang[0], ang[2], alpha=alph, c='black')
        # ax.set_xlabel("Roll, [0, pi/2]")
        # ax.set_ylabel("Yaw, [0, 2pi]")
        # plt.show()
        # return xhat, np.unique(counts).astype(np.int64)
    
    def gmm_components(self, resampled_angles):
        n_components = np.arange(1, 6)
        models = [GaussianMixture(n, random_state=0, covariance_type="full").fit(resampled_angles) for n in n_components]
        highest_model = 1
        for m, n in zip(models, n_components):
            count = 0
            for cov in m.covariances_:
                if np.any(np.diag(cov) < 0.1): # make sure gmm covs aren't too skinny
                    break
                else:
                    count += 1
            if count == n:
                highest_model = n
        
        """# log_likelihoods = [model.score(resampled_angles) for model in models]
        # x_all = np.array([n_components[0], n_components[-1]])
        # y_all = np.array([log_likelihoods[0], log_likelihoods[-1]])
        # line_coefficients = np.polyfit(x_all, y_all, 1)
        # line = np.poly1d(line_coefficients)
        # distances = np.abs(line(n_components) - log_likelihoods) / np.sqrt(1 + line_coefficients[0]**2)
        # elbow_index = np.argmax(distances) + 1
        # print("# comps:", elbow_index)
        # aic = [model.aic(resampled_angles) for model in models]
        # bic = [model.bic(resampled_angles) for model in models]
        # plt.plot(n_components, aic, label='AIC')
        # plt.show()
        # plt.close()
        # plt.plot(n_components, bic, label='BIC')
        # plt.show()
        # plt.close()
        # plt.plot(n_components, log_likelihoods, label='log-likelihood', marker='o')
        # plt.scatter(elbow_index, log_likelihoods[elbow_index], c="r")
        # plt.xlabel('Number of components')
        # plt.ylabel('Score')
        # plt.legend()
        # plt.show()
        # plt.close()"""
        return highest_model # elbow_index + 1
    
    def gmm(self, pipeline):
        pose_unc = self.uncertainty_by_pose_calc(pipeline)
        np.savetxt("points.txt", pose_unc)
        pose_unc = np.loadtxt("points.txt")
        resampled_rpy, counts = self.resampling_alg(pose_unc, resampling_criteria=10)
        # self.uncertainty_by_pose_calc(pipeline, self.hemi_points(pipeline)[counts])
        self.view_pose_unc_correlation(self.hemi_points(pipeline)[counts], pose_unc[counts, -1])
        self.view_pose_unc_correlation(self.hemi_points(pipeline), pose_unc[:, -1])
        # resampled_rpy = resampled_rpy[counts]
        n_comp = self.gmm_components(resampled_rpy)
        print("GMM comps", n_comp)
        if n_comp > 5:
            n_comp = 5
        gmm = GaussianMixture(n_components=n_comp, tol=1e-5, max_iter=1000)
        gmm.fit(resampled_rpy)
        labels = np.zeros(len(resampled_rpy))
        means = gmm.means_
        covariances = gmm.covariances_
        weights = gmm.weights_
        for i, (mean, cov, w) in enumerate(zip(means, covariances, weights)):
            labels += w * mvn.pdf(resampled_rpy, mean, cov) / len(means)
        return means, covariances, weights, labels, resampled_rpy, gmm
    
    def multi_mvn(self, x, means, covs, weights):
        val = 0.0
        for i in range(len(means)):
            val += mvn.pdf(x, means[i], covs[i]) * weights[i]
        return val
    
    def vis(self, resampled_rpy, means, covs, weights):
        samples = self.verify_so3(means, resampled_rpy, weights)
        samples = np.asarray(samples)
        if len(means.shape) > 1:
            n_mix = means.shape[0]
        else:
            n_mix = 1
            means = np.array([means])

        N = 100
        grids_r, grids_p, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, N), np.linspace(-np.pi, np.pi, N), np.linspace(-np.pi, np.pi, N)])
        grids = np.array([grids_r.ravel(), grids_p.ravel(), grids_y.ravel()]).T 
        grids = self.agent.eulerToSo3(grids)
        pdf_grids = np.zeros(len(grids))
        tgt = Distr(self.agent.eulerToSo3(means), covs, weights)
        # for i, st in tqdm(enumerate(grids), total=len(grids)):
        #     pdf_grids[i] = tgt.pdf(st)
        # pdf_grids.reshape((N, N, N))
        # vis_rp = np.hstack((pdf_grids[:, 0], pdf_grids[:, 1]))
        
        grids_r, grids_p, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids = np.array([grids_r.ravel(), grids_p.ravel(), grids_y.ravel()]).T 
        
        grids_r, grids_p = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids_rp = np.array([grids_r.ravel(), grids_p.ravel()]).T
        vis_covs = np.empty([n_mix, 2, 2])
        for i in range(n_mix):
            vis_covs[i, 0, 0] = covs[i, 0, 0]
            vis_covs[i, 0, 1] = covs[i, 0, 1]
            vis_covs[i, 1, 0] = covs[i, 1, 0]
            vis_covs[i, 1, 1] = covs[i, 1, 1]
        vis_rp = self.multi_mvn(
            grids_rp,
            np.hstack((np.expand_dims(means[:, 0], axis=1), np.expand_dims(means[:, 1], axis=1))),
            vis_covs,
            weights).reshape((100, 100))

        grids_r, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids_ry = np.array([grids_r.ravel(), grids_y.ravel()]).T
        for i in range(n_mix):
            vis_covs[i, 0, 0] = covs[i, 0, 0]
            vis_covs[i, 0, 1] = covs[i, 0, 2]
            vis_covs[i, 1, 0] = covs[i, 2, 0]
            vis_covs[i, 1, 1] = covs[i, 2, 2]
        vis_ry = self.multi_mvn(
            grids_ry,
            np.hstack((np.expand_dims(means[:, 0], axis=1), np.expand_dims(means[:, 2], axis=1))),
            vis_covs,
            weights).reshape((100, 100))
        
        grids_p, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids_py = np.array([grids_p.ravel(), grids_y.ravel()]).T
        for i in range(n_mix):
            vis_covs[i, 0, 0] = covs[i, 1, 1]
            vis_covs[i, 0, 1] = covs[i, 1, 2]
            vis_covs[i, 1, 0] = covs[i, 2, 1]
            vis_covs[i, 1, 1] = covs[i, 2, 2]
        vis_py = self.multi_mvn(
            grids_py,
            np.hstack((np.expand_dims(means[:, 1], axis=1), np.expand_dims(means[:, 2], axis=1))),
            vis_covs,
            weights).reshape((100, 100))
        
        fig, axes = plt.subplots(2, 2, figsize=(16,6), tight_layout=True)
        ax = axes[0, 0]
        ax.cla()
        ax.set_aspect('auto')
        ax.set_title("Roll vs Pitch")
        ax.contourf(grids_r, grids_y, vis_rp, cmap='Reds')
        ax.scatter(resampled_rpy[:, 0], resampled_rpy[:, 1], alpha=0.5)
        # ax.scatter(samples[:, :, 0], samples[:, :, 1], c="k", alpha=0.5)
        ax.scatter(self.plot_max[0], self.plot_max[1], c="g", alpha=1.0, s=20)

        ax = axes[1, 0]
        ax.cla()
        ax.set_title("Roll vs Yaw")
        ax.set_aspect('auto')
        ax.contourf(grids_r, grids_y, vis_ry, cmap='Reds')
        ax.scatter(resampled_rpy[:, 0], resampled_rpy[:, 2], alpha=0.5)
        # ax.scatter(samples[:, :, 0], samples[:, :, 2], c="k", alpha=0.5)
        ax.scatter(self.plot_max[0], self.plot_max[2], c="g", alpha=1.0, s=20)

        ax = axes[1, 1]
        ax.cla()
        ax.set_title("Pitch vs Yaw")
        ax.set_aspect('auto')
        ax.contourf(grids_r, grids_y, vis_py, cmap='Reds')
        ax.scatter(resampled_rpy[:, 1], resampled_rpy[:, 2], alpha=0.5)
        # ax.scatter(samples[:, :, 1], samples[:, :, 2], c="k", alpha=0.5)
        ax.scatter(self.plot_max[1], self.plot_max[2], c="g", alpha=1.0, s=20)

        plt.show()
        plt.close()

        return vis_rp, vis_ry, vis_py, grids_r, grids_p, grids_y
    
    def calc_so3_cov(self, resampled_rpy, means):
        if len(means.shape) > 1:
            n_mix = means.shape[0]
        else:
            n_mix = 1
        covs = np.zeros((n_mix, 3, 3))
        for i, mu in enumerate(means):
            x = resampled_rpy - mu
            covs[i] = np.dot(x.T, x) / (x.shape[0] - 1)
        return covs
    
    def verify_so3(self, means, resampled_rpy, weights):
        covs = self.calc_so3_cov(resampled_rpy, means)
        if len(means.shape) > 1:
            n_mix = means.shape[0]
        else:
            n_mix = 1
            means = np.array([means])
        num_samples = 150
        samples = np.zeros((n_mix, num_samples, 3))
        samples_so3 = np.zeros((n_mix, num_samples, 3, 3))
        for i, (mu, cov) in enumerate(zip(means, covs)):
            samples[i] = mvn.rvs(np.zeros_like(mu), cov, size=num_samples)
            samples_so3[i] = self.agent.eulerToSo3(samples[i])
            for k, so3 in enumerate(samples_so3[i]):
                samples_so3[i, k] = so3 @ self.agent.eulerToSo3(mu)
        
        samples = []
        for so3_list in samples_so3:
            samples.append(self.agent.stToEuler(so3_list))
        return samples

    def nbv(self, pipeline,num_views):
        means, covariances, weights, labels, resampled_rpy, gmm = self.gmm(pipeline)
        covs = self.calc_so3_cov(resampled_rpy, means)
        covs = covs.reshape(-1, 3, 3)
        covs += np.tile(np.eye(3)*1e-6, (len(covs), 1, 1))
        np.savetxt("means.txt", means)
        np.savetxt("weights.txt", weights)
        np.savetxt("covs.txt", covs.reshape(-1, 3))
        return resampled_rpy, labels, means.reshape(-1, 3), covariances, weights.reshape(-1, 1)
        print(means)
        print(weights)
        self.plot_max = resampled_xyzrpy[np.argmax(labels)]
        self.vis(resampled_xyzrpy[:, 3:], means, covariances, weights)

    def nbv_resampled(self, pipeline, num_views, curr_so3=None):
        # pose_unc = self.uncertainty_by_pose_calc(pipeline)
        # np.savetxt("erg_points.txt", pose_unc)
        pose_unc = np.loadtxt("points_entropy.txt")
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
        
        # lego, M = 15
        resampled_rpy, counts, weights = self.resampling_alg(pose_unc, resampling_criteria=15, num_views=num_views, dist=curr_so3)
        
        # ship, M = 10?
        # resampled_rpy, counts, weights = self.resampling_alg(pose_unc, resampling_criteria=11, num_views=num_views, dist=curr_so3)
        # self.view_pose_unc_correlation(self.hemi_points(pipeline)[counts], weights[counts])
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
    os.chdir("/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs")
    dir_to_save = load_config_dir + "/data/ergodic_"
    if json_file is None:
        if str(trainer.pipeline.datamanager.dataparser.data). __contains__("lego"):
            json_file = "/home/ayush/Desktop/datasets/nerf_synthetic/lego/camera_paths/synthetic_lego.json"
        elif str(trainer.pipeline.datamanager.dataparser.data). __contains__("drums"):
            json_file = "/home/ayush/Desktop/datasets/nerf_synthetic/drums/camera_paths/synthetic_drums.json"
        elif str(trainer.pipeline.datamanager.dataparser.data).split("/")[-1].__contains__("materials"):
            json_file = "/home/ayush/Desktop/datasets/nerf_synthetic/materials_scale/camera_paths/materials_synthetic.json"
        else:
            print("didn't define json file")

    os.makedirs(load_config_dir, exist_ok=True)
    os.makedirs(load_config_dir + "/data", exist_ok=True)
    os.makedirs(load_config_dir + "/data/images", exist_ok=True)

    curr_cams = np.loadtxt(str(trainer.config.get_base_dir()) + "/data/selected.txt").astype(np.int64)
    data_json = str(trainer.config.pipeline.datamanager.dataparser.data) +  "/transforms_train.json"
    meta = load_from_json(Path(data_json))
    so3_cams = np.zeros((len(curr_cams), 3, 3))
    hemi_angles = np.zeros((len(curr_cams), 3))
    for i, frame in enumerate(np.asarray(meta["frames"])[curr_cams]):
        so3_cams[i] = (np.array(frame["transform_matrix"])[:-1, :-1])
        hemi_angles[i] = (np.array(frame["hemi_angles"]))

    erg = ErgodicSelector(dir_to_save=dir_to_save, save_data=True, json_file=json_file)
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
    # candidate_cams = idx_to_cam(trainer.config.get_base_dir() + "/data/candidates.txt", trainer.pipeline.config.datamanager.data, split="train", scale_factor=trainer.pipeline.datamanager.dataparser.scale_factor)

    '''# resampled_rpy, labels, means, covs, weights = erg.nbv(trainer.pipeline, num_views=num_views)
    # print(resampled_rpy[np.flip(np.argsort(labels))[:5]])
    # means = np.loadtxt("means.txt").reshape(-1, 3)
    # covs = np.loadtxt("covs.txt").reshape(-1, 3, 3)
    # weights = np.loadtxt("weights.txt").reshape(-1, 1)
    if len(np.shape(curr_cams)) == 0:
        curr_cams = np.array([curr_cams])
    if num_views == 1:
        best_rpy = resampled_rpy[np.argmax(labels)]
        np.savetxt("rpy_traj.txt", best_rpy)
        # _, adding_index = find_closest_cam(erg.rpy_to_SE3(best_rpy), candidate_cams.camera_to_worlds.cpu().numpy(), num=1)
    else:
        kernel_objective = KernelObjective(erg.agent.eulerToSo3(means), covs, weights, num_samples=num_views, N=50, seed=0)
        optimal_kernel = kernel_objective.find_optimal()
        print("optimal kernel!!", optimal_kernel)
        optimal_kernel = 16.
        # optimal_kernel = 166.
        # if len(np.unique(resampled_rpy, axis=0)) > 10:

        with open(str(trainer.pipeline.config.datamanager.data) + "/transforms_train.json", 'r') as file:
        # with open("/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/outputs/ergodic_0/nerfacto/crop_400/data/transforms_train.json", 'r') as file:
            prev_traj = np.zeros((len(curr_cams), 3))
            data = json.load(file)
            for i, frame in enumerate(np.asarray(data["frames"])[curr_cams]):
                prev_traj[i] = frame["hemi_angles"]
        
        info_w = 25e-4 * 200 / len(curr_cams) # results with 2e-3
        st_traj = ergodic_search(erg.agent.eulerToSo3(means), covs, weights, optimal_kernel, tsteps=num_views, info_w=info_w, explr_w=10., prev_traj=erg.agent.eulerToSo3(prev_traj))
        # st_traj = ergodic_search(erg.agent.eulerToSo3(means), covs, weights, optimal_kernel, tsteps=num_views, info_w=info_w, explr_w=20., prev_traj=erg.agent.eulerToSo3(prev_traj))
        # st_traj = ergodic_search(erg.agent.eulerToSo3(means), covs, weights, optimal_kernel, tsteps=num_views, info_w=0.05, explr_w=2.)
        # else:
        #     st_traj = ergodic_search(erg.agent.eulerToSo3(means), covs, weights, optimal_kernel, tsteps=num_views, info_w=0.05, explr_w=2.5)
        np.savetxt(load_config_dir + "/data/st_traj.txt", st_traj.reshape(-1, 3))
        rpy_traj = erg.agent.stToEuler(st_traj)
        np.savetxt(load_config_dir + "/data/rpy_traj.txt", rpy_traj)'''

    start_index = 116 - trainer.config.subset_data + len(curr_cams)
    adding_index = np.arange(start_index, start_index+num_views)
    new_fp = os.path.abspath(load_config_dir) + "/data/images/transforms.json"
    # command = "cd /home/ayush/Desktop/datasets/blend_files_fixed && /home/ayush/Downloads/blender-4.1.1-linux-x64/blender -b ship_scale.blend --python 360_view.py {0} -- --cycles-device CUDA".format(os.path.abspath(load_config_dir) + "/data/images", "/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/" + load_config_dir + "/data/transforms_train.json")
    # command = "cd /home/ayush/Desktop/datasets/blend_files_fixed && /home/ayush/Downloads/blender-4.1.1-linux-x64/blender -b mic_scale.blend --python 360_view.py {0} -- --cycles-device CUDA".format(os.path.abspath(load_config_dir) + "/data/images", "/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/" + load_config_dir + "/data/transforms_train.json")
    # command = "cd /home/ayush/Desktop/datasets/blend_files_fixed && /home/ayush/Downloads/blender-4.1.1-linux-x64/blender -b materials.blend --python 360_view.py {0} -- --cycles-device CUDA".format(os.path.abspath(load_config_dir) + "/data/images", "/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/" + load_config_dir + "/data/transforms_train.json")
    # command = "cd /home/ayush/Desktop/datasets/blend_files_fixed && /home/ayush/Downloads/blender-4.1.1-linux-x64/blender -b chair.blend --python 360_view.py {0} {1} -- --cycles-device CUDA".format(os.path.abspath(load_config_dir) + "/data/images", "/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/" + load_config_dir + "/data/transforms_train.json")
    command = "cd /home/ayush/Desktop/datasets/blend_files_fixed && /home/ayush/Downloads/blender-4.1.1-linux-x64/blender -b lego.blend --python 360_view.py {0} {1} -- --cycles-device CUDA".format(os.path.abspath(load_config_dir) + "/data/images", "/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/" + load_config_dir + "/data/transforms_train.json")
    # command = "cd /home/ayush/Desktop/datasets/blend_files_fixed && /home/ayush/Downloads/blender-2.80-linux-glibc217-x86_64/blender -b lego.blend --python 360_view.py {0} {1}".format("/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/" + load_config_dir + "/data/images", "/home/ayush/Desktop/nerf/ros_nerf/src/ros_nerf/uncertainty_NeRFs/" + load_config_dir + "/data/transforms_train.json")
    sp.call(command, shell=True)

    """for i, rpy in enumerate(rpy_traj):
        if np.abs(rpy[0]) > np.pi/2:
            print(i, "not hemisphere?")
        # points
        xyz = spherical_to_hemi(rpy_traj[:, 2::-2], erg.radius, erg.height, origin=erg.ground_position)
        c2ws = erg.generate_c2ws(rpy_traj[:, 2::-2], xyz)
        # c2ws = np.concatenate((st_traj, np.expand_dims(xyz, axis=2)), axis=2)
        cam_SE3 = np.zeros((num_views, 4, 4))
        for i, (st, point) in enumerate(zip(st_traj, xyz)):
            cam_SE3[i] = np.vstack((np.hstack((st, point.reshape(-1, 1))), np.zeros((1, 4))))
        cam_SE3[:, -1, -1] = 1.0

        resolution = np.array([400, 400])
        fov = 75.0
        fx = three_js_perspective_camera_focal_length(fov, resolution[0])
        fy = fx
        cx = resolution[1] / 2
        cy = resolution[0] / 2
        mask_val = 0.2
        unc_value = np.zeros(len(c2ws))
        pipeline = trainer.pipeline

        test_cam = Cameras(fx=torch.tensor(fx), 
                        fy=torch.tensor(fy), 
                        cx=torch.tensor(cx),
                        cy=torch.tensor(cy), 
                        camera_type=torch.tensor([1]),
                        camera_to_worlds=torch.zeros((3, 4)))
        for i, pose in tqdm(enumerate(c2ws), desc="Calculating uncertainty values", total=len(c2ws)):
            pipeline.eval()
            r = np.linalg.norm(pose[:, -1].flatten() - erg.center_of_mass)**2
            middle = None # 250
            test_cam.camera_to_worlds = pose
            with background_color_override_context(erg.background), torch.no_grad():
                outs = pipeline._model.get_outputs_for_camera(test_cam, obb_box=erg.obb, middle=middle)
            # outs["rgb_std_dev"] = (outs["rgb_std_dev"] - torch.min(outs["rgb_std_dev"]))/(torch.max(outs["rgb_std_dev"]) - torch.min(outs["rgb_std_dev"]))
            # outs['rgb_std_dev'].masked_fill_((torch.logical_and(outs["rgb_std_dev"] < 1e-5, outs['rgb'] > .99)), 1.0)
            # outs['rgb_std_dev'] = torch.sum(outs['rgb_std_dev'], axis=-1)/3
            # mask = (outs['rgb_std_dev'] < mask_val).clone()
            # outs['rgb_std_dev'].masked_fill_(mask, 0.0)
            # # percent_in_image = (np.count_nonzero(outs["rgb"].numpy()) / np.prod((outs['rgb'].numpy().shape)))
            # # unc_value[i] = np.sum(outs["rgb_std_dev"].numpy())/np.prod((outs['rgb_std_dev'].numpy().shape)) * (-.75*np.tanh((2.5*percent_in_image - 1)) + 1)
            # unc_value[i] = np.sum(outs["rgb_std_dev"].numpy())/np.prod((outs['rgb_std_dev'].numpy().shape)) * (.75*np.tanh((1/erg.radius*(r - erg.scale_r) - 1)) + 1)
            # print(r, unc_value[i])
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(outs['rgb'])
            # ax[1].imshow(outs['rgb_std_dev'])
            plt.show()
            plt.close()"""
        
        # adding_index = []
        # candidate_cams = candidate_cams.camera_to_worlds.cpu().numpy()
        # for cam in cam_SE3:
        #     _, index = find_closest_cam(cam, candidate_cams, num=1)
        #     adding_index.append(index)
        #     candidate_cams = np.delete(candidate_cams, index, axis=0)
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
    parser = argparse.ArgumentParser(description='Args necessary for ergodic selector')
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
    print(run(trainer, args.dir_to_save, json_file = "/home/ayush/Desktop/datasets/nerf_synthetic/chair_new/camera_paths/chair_synthetic.json", num_views=6))
    # print(run(trainer, args.dir_to_save, json_file = "/home/ayush/Desktop/datasets/nerf_synthetic/lego/camera_paths/synthetic_lego.json", num_views=8))

    # print(run(trainer, args.load_config_dir[:-10], json_file = "/home/ayush/Desktop/datasets/nerf_synthetic/lego/camera_paths/synthetic_lego.json", num_views=8))