import matplotlib.pyplot as plt 
from tqdm import tqdm 
import modern_robotics as mr
import numpy as np
import os
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal as mvn

from .so3_cost import quadratic_cost, d1_quad_cost, d_exp_mat

#####################################################
# GMM - Euclidean Space
#####################################################
class GMM_Euclidean:
    def __init__(self, means, covs, ws):
        # means = np.array([means])
        assert len(means) == len(covs)
        assert len(means) == len(ws)
        
        self.nmix = len(means)
        self.dim = len(means[0])

        self.means = np.array(means)
        self.covs = np.array(covs)
        self.ws = np.array(ws)

        self.covs_inv = []
        for _cov in self.covs:
            self.covs_inv.append(np.linalg.inv(_cov))
        self.covs_inv = np.array(self.covs_inv)

        self.norm = self.get_norm()
    
    def get_norm(self):
        norm_val = 0.0

        for i in range(self.nmix):
            for j in range(self.nmix):
                mean_i = self.means[i]
                mean_j = self.means[j]
                cov_i = self.covs[i]
                cov_j = self.covs[j]
                w_i = self.ws[i]
                w_j = self.ws[j]
                
                prefix = 1.0 / np.sqrt(np.linalg.det(2 * np.pi * (cov_i + cov_j))) * w_i * w_j
                norm_val_ij = prefix * np.exp(-0.5 * np.linalg.inv(cov_i + cov_j) @ (mean_i - mean_j) @ (mean_i - mean_j)) 
                norm_val += norm_val_ij
        
        return norm_val
    
    def pdf(self, x):
        val = 0.0
        for i in range(self.nmix):
            val += mvn.pdf(x, self.means[i], self.covs[i]) * self.ws[i]
        return val 
    
    def dpdf(self, x):
        dvec = 0.0
        for i in range(self.nmix):
            dvec += -1.0 * (self.covs_inv[i] @ (x - self.means[i])) * mvn.pdf(x, self.means[i], self.covs[i]) * self.ws[i]
        return dvec

#####################################################
# Distribution Implementation - specifically for SO(3)
# following https://arxiv.org/pdf/2403.01536
#####################################################
class Distr:
    def __init__(self, mean_list, cov_list, w_list):
        assert len(mean_list) == len(cov_list)
        assert len(mean_list) == len(w_list)
        assert len(mean_list) > 0

        self.mean_list = mean_list
        self.cov_list = cov_list
        self.n_modes: int = len(mean_list)
        self.w_list = w_list
        assert np.abs(np.sum(self.w_list) - 1.0) < 1e-4

        self.cov_inv_list = []
        self.eta_list = []
        for sig in self.cov_list:
            # self.cov_inv_list.append(np.linalg.inv(sig))
            self.cov_inv_list.append(np.linalg.pinv(sig, rcond=1e-3))
            self.eta_list.append(1.0 / np.sqrt(np.linalg.det(sig) * (2*np.pi)**3))

    def normal_pdf(self, st, mean, eta, cov_inv) -> float:
        # Eq. 63
        val = np.exp(-0.5 * quadratic_cost(st, mean, cov_inv))
        # x = Rotation.from_matrix(st).as_euler('XYZ', degrees=False)[0]
        # val *= (x < np.pi/2 and x > 0)
        return eta * val
    
    def dnormal_pdf(self, st, mean, eta, cov_inv) -> np.ndarray: # returns 3x1 vector
        # Eq. 65
        prob = self.normal_pdf(st, mean, eta, cov_inv)
        d_dg = d_exp_mat(st, mean, cov_inv)
        log_g_ginv = np.expand_dims(mr.so3ToVec(mr.MatrixLog3(mr.RotInv(mean) @ st)), axis=1)
        return -0.5 * prob * (d_dg.T) @ cov_inv @ log_g_ginv

    def pdf(self, st) -> float: # st is type SO(3), i.e. 3x3
        val = 0.0
        for mu, eta, cov_inv, w in zip(self.mean_list, self.eta_list, self.cov_inv_list, self.w_list):
            val += w * self.normal_pdf(st, mu, eta, cov_inv)
        return val
    
    def dpdf(self, st) -> np.ndarray: # st is type SO(3), i.e. 3x3
        val = np.zeros((3, 1))
        for mu, eta, cov_inv, w in zip(self.mean_list, self.eta_list, self.cov_inv_list, self.w_list):
            val += w * self.dnormal_pdf(st, mu, eta, cov_inv)
        return val


#####################################################
# Kernel SO(3)
#####################################################
class KernelSO3:
    def __init__(self, theta):
        self.theta = theta
        self.cov = np.eye(3) / theta
        # self.cov[1, 1] = 1e5
        self.cov_inv = np.linalg.pinv(self.cov, rcond=1e-4)
        self.eta = 1.0 / (np.sqrt((2*np.pi)**3 * np.linalg.det(self.cov)))

    def eval(self, x1, x2) -> float:
        return self.eta * np.exp(-0.5 * quadratic_cost(x1, x2, self.cov_inv))
    
    def d1_eval(self, x1, x2) -> np.ndarray:
        return -0.5 * self.eval(x1, x2) * d1_quad_cost(x1, x2, self.cov_inv)


class KernelErgodicCost:
    def __init__(self, 
                 straj,
                 kernel: KernelSO3, 
                 tgt: Distr,
                 info_gain=1.0, 
                 explr_gain=1.0, 
                 prev_traj=None):
        self.straj = straj
        self.kernel = kernel
        self.N = len(self.straj)
        self.tgt = tgt
        self.info_gain, self.expr_gain = info_gain, explr_gain
        self.prev_traj = prev_traj

        if prev_traj is not None:
            self.prev_info_loss = 0.0
            for st in self.prev_traj:
                self.prev_info_loss += self.tgt.pdf(st)
            self.prev_info_loss *= -1.0 * self.info_gain / len(self.prev_traj)
            
            self.prev_explr_loss = 0.0
            for st1 in self.prev_traj:
                for st2 in self.prev_traj:
                    self.prev_explr_loss += self.kernel.eval(st1, st2)
            self.prev_explr_loss *= self.expr_gain / (len(self.prev_traj))**2

            self.prev_explr_dvec = np.zeros((3, 1))
            for s_curr in self.prev_traj:
                self.prev_explr_dvec += self.kernel.d1_eval(st, s_curr)
            self.prev_explr_dvec *= self.expr_gain / (len(self.prev_traj))**2
    
    def cost(self):
        info_cost = 0
        for st in self.straj:
            info_cost += -self.tgt.pdf(st)
        info_cost *= -1.0 * self.info_gain / self.N

        explr_cost = 0
        for st1 in self.straj:
            for st2 in self.straj:
                explr_cost += self.kernel.eval(st1, st2)
        explr_cost *= self.expr_gain / self.N**2
        
        if self.prev_traj is not None:
            info_cost += self.prev_info_loss
            explr_cost + self.prev_explr_loss

        return info_cost + explr_cost
    
    def dcost(self):
        info_dvec = np.zeros((self.N, 3, 1))
        for i, st in enumerate(self.straj):
            dinfo_cost = np.zeros((3, 1))
            dinfo_cost += -self.tgt.dpdf(st)
            dinfo_cost *= self.info_gain / self.N

            dexplr_cost = np.zeros((3, 1))
            for s_curr in self.straj:
                dexplr_cost += self.kernel.d1_eval(st, s_curr)
            dexplr_cost *= self.expr_gain / self.N**2
            if self.prev_traj is not None:
                dexplr_cost += self.prev_explr_dvec
            info_dvec[i] = dinfo_cost + dexplr_cost
        return info_dvec
        
    def find_optimal(self):
        param_opts = np.power(10.0, np.arange(-5, 5))
        obj_val = np.zeros(len(param_opts))
        for i, param in tqdm(enumerate(param_opts), total=len(param_opts)):
            self.kernel = KernelSO3(param)
            obj_val[i] = self.cost()
        print("coarse", param_opts[np.argmin(obj_val)])
        param_opts = np.linspace(1, 100, 150) * param_opts[np.argmin(obj_val)]
        obj_val = np.zeros(len(param_opts))
        for i, param in tqdm(enumerate(param_opts), total=len(param_opts)):
            self.kernel = KernelSO3(param)
            obj_val[i] = self.cost()
        return param_opts[np.argmin(obj_val)]


def kernel_search(means, covs, weights, optimal_kernel, Q_diag=1., R_diag=1., info_w=1e-2, explr_w=5e-1, ctrl_w=0.0, dt=0.1, dim=3, tsteps=100, prev_traj=None):
    agent = Agent(dt = dt, dim = dim)
    vis_means = agent.stToEuler(means)
    tgt = Distr(means, covs, weights)
    Q = np.eye(dim) * Q_diag
    Q[0, 0] *= 1e-2
    Q[1, 1] *= 100.
    R = np.eye(dim) * R_diag
    kernel = KernelSO3(theta = optimal_kernel)
    print("prev_traj", prev_traj.shape)
    ilqr = SO3_LQR(
        tsteps=tsteps,
        dt=dt,
        agent=agent,
        tgt=tgt,
        kernel=kernel,
        Q=Q, R=R,
        info_w=info_w,
        explr_w=explr_w,
        ctrl_w=ctrl_w,
        barr_w=0.,
        prev_traj=prev_traj
        )
    if optimal_kernel < 10:
        ilqr.set_info_w(1e-4)
        ilqr.set_explr_w(4e-1)
    elif optimal_kernel > 200:
        ilqr.set_explr_w(5e-2)
        if optimal_kernel > 1e4:
            ilqr.set_info_w(5e-3)
    show_plots = True
    
    # from .optimize_params import SO3World
    # from .bootstrap import TSP
    # world = SO3World(num_samples=tsteps, N=50)
    # world.load_distribution(means, covs, weights)
    # points = world.target_dist_samples()
    # tsp = TSP(points)
    # target_points = tsp.opt_2_algorithm()
    # tsp_points = agent.stToEuler(target_points)
    # s0 = target_points[0]
    # target_points = Rotation.from_matrix(target_points)
    # controls = np.zeros((tsteps, dim, 1))
    # current_rotation = Rotation.from_matrix(s0)
    # for i, desired_rotation in enumerate(target_points):
    #     controls[i] = np.expand_dims(agent.inv_step(desired_rotation.as_matrix(), current_rotation.as_matrix()), axis=1)
    #     current_rotation = desired_rotation
    # u_traj = controls.copy()
    
    s0 = vis_means[np.argmax(weights)].copy()
    # s0[0] = 0.
    # s0[2] = 0.
    # s0[2] -= 1.2
    s0 = agent.eulerToSo3(s0)
    u_traj = np.zeros((tsteps, dim, 1)) + 0.1
    # u_traj = np.expand_dims(np.tile(np.array([0.05, 0.0, 0.5]), (tsteps, 1)), axis=2) / dt
    u_traj = np.expand_dims(np.tile(np.array([-0.025, 0.0, -1.0 * np.sign(vis_means[np.argmax(weights)][2]) * 0.25]), (tsteps, 1)), axis=2) / dt
    # u_traj = np.expand_dims(np.tile(np.array([0.05, 0.0, 0.75]), (tsteps//2, 1)), axis=2) / dt
    # u_traj2 = np.expand_dims(np.tile(np.array([0.02, 0.0, -0.5]), (tsteps//2 + tsteps%2, 1)), axis=2) / dt
    # u_traj = np.concatenate((u_traj, u_traj2))

    #################################################
    # Visualization
    #################################################
    def vis():
        grids_r, grids_p, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids = np.array([grids_r.ravel(), grids_p.ravel(), grids_y.ravel()]).T 
        
        grids_r, grids_p = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids_rp = np.array([grids_r.ravel(), grids_p.ravel()]).T
        vis_covs = np.empty([len(means), 2, 2])
        for i in range(len(means)):
            vis_covs[i, 0, 0] = covs[i, 0, 0]
            vis_covs[i, 0, 1] = covs[i, 0, 1]
            vis_covs[i, 1, 0] = covs[i, 1, 0]
            vis_covs[i, 1, 1] = covs[i, 1, 1]
        
        vis_rp = GMM_Euclidean(
            np.hstack(
                (np.expand_dims(vis_means[:, 0], axis=1),
                np.expand_dims(vis_means[:, 1], axis=1))),
            vis_covs, np.ones(len(means))/len(means)).pdf(
                                        grids_rp).reshape((100, 100))

        grids_r, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids_ry = np.array([grids_r.ravel(), grids_y.ravel()]).T
        for i in range(len(means)):
            vis_covs[i, 0, 0] = covs[i, 0, 0]
            vis_covs[i, 0, 1] = covs[i, 0, 2]
            vis_covs[i, 1, 0] = covs[i, 2, 0]
            vis_covs[i, 1, 1] = covs[i, 2, 2]
        vis_ry = GMM_Euclidean(
            np.hstack(
                (np.expand_dims(vis_means[:, 0], axis=1),
                np.expand_dims(vis_means[:, 2], axis=1))),
            vis_covs, np.ones(len(means))/len(means)).pdf(
                                        grids_ry).reshape((100, 100))

        grids_p, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids_py = np.array([grids_p.ravel(), grids_y.ravel()]).T
        for i in range(len(means)):
            vis_covs[i, 0, 0] = covs[i, 1, 1]
            vis_covs[i, 0, 1] = covs[i, 1, 2]
            vis_covs[i, 1, 0] = covs[i, 2, 1]
            vis_covs[i, 1, 1] = covs[i, 2, 2]
        vis_py = GMM_Euclidean(
            np.hstack(
                (np.expand_dims(vis_means[:, 1], axis=1),
                np.expand_dims(vis_means[:, 2], axis=1))),
            vis_covs, np.ones(len(means))/len(means)).pdf(
                                        grids_py).reshape((100, 100))

        return vis_rp, vis_ry, vis_py, grids_r, grids_p, grids_y
    
    vis_rp, vis_ry, vis_py, grids_r, grids_p, grids_y = vis()
    # fig, ax = plt.subplots(1, 3)
    # axis = [0, 1]
    # ax[axis[0]].contourf(grids_r, grids_y, vis_rp, cmap='Reds')
    # ax[axis[0]].scatter(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # # ax[axis[0]].plot(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # # ax[axis[0]].plot(original_points[:, axis[0]], original_points[:, axis[1]], linestyle="--", c="k")
    # ax[axis[0]].set_title("Roll vs Pitch")

    # axis = [0, 2]
    # ax[1].contourf(grids_r, grids_y, vis_ry, cmap='Reds')
    # ax[1].scatter(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # # ax[1].plot(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # # ax[1].plot(original_points[:, axis[0]], original_points[:, axis[1]], linestyle="--", c="k")
    # ax[1].set_title("Roll vs Yaw")

    # axis = [1, 2]
    # ax[2].contourf(grids_r, grids_y, vis_py, cmap='Reds')
    # ax[axis[1]].scatter(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # # ax[axis[1]].plot(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # # ax[axis[1]].plot(original_points[:, axis[0]], original_points[:, axis[1]], linestyle="--", c="k")
    # ax[axis[1]].set_title("Pitch vs Yaw")
    
    # plt.show()
    # plt.close()
    if show_plots:
        fig, axes = plt.subplots(1, 3, figsize=(16,6), tight_layout=True)
    total_steps = 500

    for iter in range(500):
        v_traj = ilqr.get_v_traj(s0, u_traj)
        step, opt_s_traj, opt_loss = ilqr.line_search(s0, u_traj, v_traj)
        u_traj += step * v_traj

        # print('loss: ', opt_loss)
        # print('step: ', step)
        if (iter-2) % 50 == 0 or step == 0.0:
            if show_plots:
                fig.suptitle(f'Iter: {iter}; Vis in Euclidean Space')
                straj_vis = agent.stToEuler(opt_s_traj)
                prev_traj_vis = agent.stToEuler(prev_traj)
                
                ax = axes[0]
                ax.cla()
                ax.set_aspect('auto')
                ax.set_title("Roll vs Pitch")
                ax.contourf(grids_r, grids_y, vis_rp, cmap='Reds')
                ax.plot(straj_vis[:,0], straj_vis[:,1], linestyle=" ", marker='o', color='k', markersize=3, alpha=0.5)
                ax.plot(prev_traj_vis[:,0], prev_traj_vis[:,1], linestyle=" ", marker='x', color='b', markersize=3, alpha=0.5)

                ax = axes[1]
                ax.cla()
                ax.set_title("Roll vs Yaw")
                ax.set_aspect('auto')
                ax.contourf(grids_r, grids_y, vis_ry, cmap='Reds')
                ax.plot(straj_vis[:,0], straj_vis[:,2], linestyle=" ", marker='o', color='k', markersize=3, alpha=0.5)
                ax.plot(prev_traj_vis[:,0], prev_traj_vis[:,2], linestyle=" ", marker='x', color='b', markersize=3, alpha=0.5)

                ax = axes[2]
                ax.cla()
                ax.set_title("Pitch vs Yaw")
                ax.set_aspect('auto')
                ax.contourf(grids_r, grids_y, vis_py, cmap='Reds')
                ax.plot(straj_vis[:,1], straj_vis[:,2], linestyle=" ", marker='o', color='k', markersize=3, alpha=0.5)
                ax.plot(prev_traj_vis[:,1], prev_traj_vis[:,2], linestyle=" ", marker='x', color='b', markersize=3, alpha=0.5)

                plt.pause(0.01)
        
        if step <= 1e-4 and iter > 2:
            # print("step is 0")
            total_steps = iter
            break
            
    if show_plots:
        plt.savefig('rpy_traj.png')
        plt.show()
        plt.close()
    print("total steps", total_steps)

    return opt_s_traj

if __name__ == "__main__":
    main()
    # agent = Agent()
    # means = agent.eulerToSo3(np.loadtxt("means.txt")).reshape(-1, 3, 3)
    # covs = np.loadtxt("covs.txt").reshape(-1, 3, 3)
    # weights = np.loadtxt("weights.txt").reshape(-1, 1)
    # st_traj = ergodic_search(means, covs, weights, 164.4295302013422, tsteps=5)