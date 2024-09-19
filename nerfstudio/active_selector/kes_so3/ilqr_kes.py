import matplotlib.pyplot as plt 
from tqdm import tqdm 
import modern_robotics as mr
import numpy as np
import os
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal as mvn

from .so3_cost import quadratic_cost, d1_quad_cost, d_exp_mat

#####################################################
# Agent
#####################################################
class Agent:
    def __init__(self, dt: float = 0.1, dim: int = 3):
        self.dt = dt
        self.dim = dim

    def dyn(self, st, ut):
        return ut.flatten()
    
    def step(self, st, ut):
        control_so3 = self.eulerToSo3(self.dt * self.dyn(st, ut))
        st_new = st @ control_so3
        st_euler = self.stToEuler(st_new)
        st_euler[1] = 0.
        if st_euler[0] > np.pi/2:
            st_euler[0] = np.pi/2
        elif st_euler[0] < 0:
            st_euler[0] = 0.
        st_new = self.eulerToSo3(st_euler)
        return st_new

    def get_A(self, st, ut):
        # -1.0 * adjoint
        return -1.0 * self.eulerToSo3(self.dyn(st, ut))
    
    def get_B(self, st, ut):
        # Identity
        return np.eye(self.dim)
    
    def traj_sim(self, s0, u_traj):
        assert len(s0) == self.dim
        
        tsteps = len(u_traj)
        traj = np.zeros((tsteps, self.dim, self.dim)) 
        st = s0.copy()
        for t in range(0, tsteps):
            ut = u_traj[t]
            st = self.step(st, ut)
            traj[t] = st.copy()

        return traj
    
    def stToEuler(self, st):
        return Rotation.from_matrix(st).as_euler('XYZ', degrees=False)
        # return mr.so3ToVec(mr.MatrixLog3(st))
    
    def eulerToSo3(self, rpy):
        return Rotation.from_euler('XYZ', rpy, degrees=False).as_matrix()
        # return mr.MatrixExp3(mr.VecToso3(rpy))

    def inv_step(self, st_curr, st_next) -> np.ndarray:
        control_so3 = np.linalg.inv(st_curr) @ st_next
        ut = mr.so3ToVec(mr.MatrixLog3(control_so3)) / self.dt
        return ut
    
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

#####################################################
# Controller
#####################################################
class SO3_LQR:
    def __init__(self, 
                 tsteps: int, 
                 dt: float, 
                 agent: Agent,
                 tgt: Distr,
                 kernel: KernelSO3,
                 Q: np.ndarray, 
                 R: np.ndarray,
                 info_w: float = 1.0,
                 explr_w: float = 1.0,
                 ctrl_w: float = 0.1,
                 barr_w: float = 0.1,
                 prev_traj: np.ndarray = None) -> None:
        self.tsteps = tsteps
        self.dt = dt
        self.dim = 3

        self.agent = agent
        self.tgt = tgt
        self.kernel = kernel

        self.Q = Q
        self.Q_inv = np.linalg.inv(Q)
        self.R = R 
        self.R_inv = np.linalg.inv(R)

        self.T = self.tsteps * self.dt 

        self.info_w = info_w
        self.ctrl_w = ctrl_w
        self.explr_w = explr_w
        self.barr_w = barr_w

        # hemisphere bound - roll shouldn't be more that np.pi/2 or less than -np.pi/2 to be on hemisphere
        self.barr_lower = np.array([[1., 0., 0.],
                                    [0., 0., 1.],
                                    [0., -1., 0.]])
        self.barr_upper = np.array([[1., 0., 0.],
                                    [0., 0., -1.],
                                    [0., 1., 0.]])
        # self.barr_M = np.zeros((3, 3))
        # self.barr_M[1, 1] = 1.0
        # self.barr_M[1, 1] = 1.0
        self.barr_M = np.eye(3)

        if prev_traj is not None:
            self.prev_traj = prev_traj
            self.prev_info_loss = 0.0
            for st in self.prev_traj:
                self.prev_info_loss += self.dt * self.tgt.pdf(st)
            self.prev_info_loss /= len(self.prev_traj) * self.dt
            
            self.prev_explr_loss = 0.0
            for st1 in self.prev_traj:
                for st2 in self.prev_traj:
                    self.prev_explr_loss += self.dt * self.dt * self.kernel.eval(st1, st2)
            self.prev_explr_loss *= self.explr_w / (len(self.prev_traj) * self.dt)**2

            self.prev_explr_dvec = np.zeros((3, 1))
            for s_curr in self.prev_traj:
                self.prev_explr_dvec += self.kernel.d1_eval(st, s_curr)
            self.prev_explr_dvec *= self.explr_w / (len(self.prev_traj) * self.dt)**2
            
        else:
            self.prev_traj = None

    def loss(self, s_traj, u_traj):
        assert len(s_traj) == self.tsteps
        assert len(u_traj) == self.tsteps

        info_loss = 0.0
        if self.info_w > 0:
            for st in s_traj:
                info_loss += self.dt * self.tgt.pdf(st)
            info_loss /= self.T
            if self.prev_traj is not None:
                info_loss += self.prev_info_loss
            info_loss *= -1.0 * 2 * self.info_w

        explr_loss = 0.0
        if self.explr_w > 0:
            for st1 in s_traj:
                for st2 in s_traj:
                    explr_loss += self.dt * self.dt * self.kernel.eval(st1, st2)
            explr_loss *= self.explr_w / self.T / self.T
            if self.prev_traj is not None:
                explr_loss += self.prev_explr_loss

        ctrl_loss = 0.0
        ctrl_loss = np.sum(np.power(np.linalg.norm(u_traj, axis=1), 2))
        ctrl_loss *= self.dt / self.T * self.ctrl_w

        barr_loss = self.barr_w * self.barr(s_traj) * self.dt / self.T

        # print("info", info_loss, "explr", explr_loss, "ctrl", ctrl_loss, "barr", barr_loss)
        return info_loss + explr_loss + ctrl_loss + barr_loss
    
    def dldx(self, st, ut, s_traj):
        info_dvec = np.zeros((3, 1))
        if self.info_w > 0.0:
            info_dvec = -2.0 * self.info_w * self.tgt.dpdf(st) / self.T
        # info_dvec[0] *= 0.5
        
        explr_dvec = np.zeros((3, 1))
        if self.explr_w > 0.0:
            for s_curr in s_traj:
                explr_dvec += self.kernel.d1_eval(st, s_curr)
            explr_dvec *= self.explr_w / self.T / self.T
            if self.prev_traj is not None:
                explr_dvec += self.prev_explr_dvec
        # explr_dvec[0] *= 2
        
        barr_vec = self.barr_w * self.barr_grad(st) / self.T
        dvec = info_dvec + explr_dvec + barr_vec
        return dvec
    
    def barr(self, s_traj):
        barr_cost = 0.0
        rpy = self.agent.stToEuler(s_traj)
        for x in rpy[:, 0]:
            barr_cost += np.sum((x > np.pi/2) * np.square(x - np.pi/2))
            # barr_cost += np.sum((x < -np.pi/2) * np.square(-np.pi/2 - x))
            barr_cost += np.sum((x < 0.) * np.square(-x))
        # for x in rpy[:, 1]:
        #     barr_cost += np.sum((np.abs(x) > 5e-2) * np.square(x - 0.))
        # for st in s_traj:
        #     pitch = -np.arcsin(st[2, 0])
        #     cos_roll = (st[2, 2] / np.cos(pitch))
        #     sin_roll = (st[2, 1] / np.cos(pitch))
        #     if cos_roll < 0: # and sin_roll > 0: # the viewing angle (roll) is outside the bounds of the hemisphere
        #         barr_cost += self.barr_w * quadratic_cost(st, self.barr_upper, self.barr_M)
        #         # np.min([quadratic_cost(st, self.barr_lower, self.barr_M),
        #         #         quadratic_cost(st, self.barr_upper, self.barr_M)])
        #         # if (np.arccos(cos_roll)) > np.pi/2:
        #         #     barr_cost += self.barr_w * quadratic_cost(st, self.barr_upper, self.barr_M)
        #         # else:
        #         #     barr_cost += self.barr_w * -1 * quadratic_cost(st, self.barr_lower, self.barr_M)
        return barr_cost
    
    def barr_grad(self, st):
        b_grad = np.zeros((3, 1))
        x = self.agent.stToEuler(st)[0]
        b_grad[0] += 2.0 * (x > np.pi/2) * (x - np.pi/2)
        # b_grad[0] += 2.0 * (x < -np.pi/2) * (-np.pi/2 - x)
        b_grad[0] += 2.0 * (x < 0) * (-x)
        # y = self.agent.stToEuler(st)[1]
        # b_grad[1] += 2.0 * (np.abs(y) > 5e-2) * (y - 0.)
        # pitch = -np.arcsin(st[2, 0])
        # cos_roll = (st[2, 2] / np.cos(pitch))
        # sin_roll = (st[2, 1] / np.cos(pitch))
        # if cos_roll < 0: # and sin_roll > 0: # the viewing angle is outside the bounds of the hemisphere
        #     b_grad += self.barr_w * d1_quad_cost(st, self.barr_upper, self.barr_M)
        #     # b_grad += self.barr_w * np.min([d1_quad_cost(st, self.barr_lower, self.barr_M),
        #     #                                 d1_quad_cost(st, self.barr_upper, self.barr_M)])
        #     b_grad[1:] = 0.
        #     b_grad *= 10
        return b_grad
    
    def dldu(self, st, ut, s_traj):
        return 2.0 * ut / self.T * self.ctrl_w
    
    def set_ctrl_w(self, w):
        self.ctrl_w = w

    def set_info_w(self, w):
        self.info_w = w
    
    def set_explr_w(self, w):
        self.explr_w = w

    def set_barr_w(self, w):
        self.barr_w = w
    
    def P_dyn_rev(self, Pt, At, Bt, at, bt):
        return Pt @ At + At.T @ Pt - Pt @ Bt @ self.R_inv @ Bt.T @ Pt + self.Q 
    
    def P_dyn_step(self, Pt, At, Bt, at, bt):
        k1 = self.dt * self.P_dyn_rev(Pt, At, Bt, at, bt)
        k2 = self.dt * self.P_dyn_rev(Pt+k1/2, At, Bt, at, bt)
        k3 = self.dt * self.P_dyn_rev(Pt+k2/2, At, Bt, at, bt)
        k4 = self.dt * self.P_dyn_rev(Pt+k3, At, Bt, at, bt)

        Pt_new = Pt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return Pt_new 
    
    def P_traj_revsim(self, PT, A_list, B_list, a_list, b_list):
        P_traj_rev = np.zeros((self.tsteps, self.dim, self.dim))
        P_curr = PT.copy()
        for t in range(self.tsteps):
            At = A_list[-1-t]
            Bt = B_list[-1-t]
            at = a_list[-1-t]
            bt = b_list[-1-t]

            P_new = self.P_dyn_step(P_curr, At, Bt, at, bt)
            P_traj_rev[t] = P_new.copy()
            P_curr = P_new 
        
        return P_traj_rev

    def r_dyn_rev(self, rt, Pt, At, Bt, at, bt):
        return (At - Bt @ self.R_inv @ Bt.T @ Pt).T @ rt + at - Pt @ Bt @ self.R_inv @ bt

    def r_dyn_step(self, rt, Pt, At, Bt, at, bt):
        k1 = self.dt * self.r_dyn_rev(rt, Pt, At, Bt, at, bt)
        k2 = self.dt * self.r_dyn_rev(rt+k1/2, Pt, At, Bt, at, bt)
        k3 = self.dt * self.r_dyn_rev(rt+k2/2, Pt, At, Bt, at, bt)
        k4 = self.dt * self.r_dyn_rev(rt+k3, Pt, At, Bt, at, bt)

        rt_new = rt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return rt_new
    
    def r_traj_revsim(self, rT, P_list, A_list, B_list, a_list, b_list):
        r_traj_rev = np.zeros((self.tsteps, self.dim, 1))
        r_curr = rT
        for t in range(self.tsteps):
            Pt = P_list[-1-t]
            At = A_list[-1-t]
            Bt = B_list[-1-t]
            at = a_list[-1-t]
            bt = b_list[-1-t]

            r_new = self.r_dyn_step(r_curr, Pt, At, Bt, at, bt)
            r_traj_rev[t] = r_new.copy()
            r_curr = r_new 

        return r_traj_rev

    def z_dyn(self, zt, Pt, rt, At, Bt, bt):
        return At @ zt + Bt @ self.z2v(zt, Pt, rt, Bt, bt)
    
    def z_dyn_step(self, zt, Pt, rt, At, Bt, bt):
        k1 = self.dt * self.z_dyn(zt, Pt, rt, At, Bt, bt)
        k2 = self.dt * self.z_dyn(zt+k1/2, Pt, rt, At, Bt, bt)
        k3 = self.dt * self.z_dyn(zt+k2/2, Pt, rt, At, Bt, bt)
        k4 = self.dt * self.z_dyn(zt+k3, Pt, rt, At, Bt, bt)

        zt_new = zt + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0 
        return zt_new

    def z_traj_sim(self, z0, P_list, r_list, A_list, B_list, b_list):
        # why are these values so small?
        z_traj = np.zeros((self.tsteps, self.dim, 1))
        z_curr = z0.copy()

        for t in range(self.tsteps):
            Pt = P_list[t]
            rt = r_list[t]
            At = A_list[t]
            Bt = B_list[t]
            bt = b_list[t]

            z_new = self.z_dyn_step(z_curr, Pt, rt, At, Bt, bt)
            z_traj[t] = z_new.copy()
            z_curr = z_new
        
        return z_traj
    
    def z2v(self, zt, Pt, rt, Bt, bt):
        return -self.R_inv @ Bt.T @ Pt @ zt - self.R_inv @ Bt.T @ rt - self.R_inv @ bt
    
    def get_v_traj(self, s0, u_traj):
        s_traj = self.agent.traj_sim(s0, u_traj)

        A_list = np.zeros((self.tsteps, self.dim, self.dim))
        # B_list = np.zeros((self.tsteps, self.dim, self.dim))
        B_list = np.tile(np.eye(self.dim), (self.tsteps, 1, 1))
        a_list = np.zeros((self.tsteps, self.dim, 1))
        b_list = np.zeros((self.tsteps, self.dim, 1))
        for _i, (_st, _ut) in enumerate(zip(s_traj, u_traj)):
            A_list[_i] = self.agent.get_A(_st, _ut)
            # B_list[_i] = self.agent.get_B(_st, _ut)
            a_list[_i] = self.dldx(_st, _ut, s_traj)
            b_list[_i] = self.dldu(_st, _ut, s_traj)

        PT = np.zeros((self.dim, self.dim))
        P_traj_rev = self.P_traj_revsim(PT, A_list, B_list, a_list, b_list)
        P_list = np.flip(P_traj_rev, axis=0)

        rT = np.zeros((self.dim, 1))
        r_traj_rev = self.r_traj_revsim(rT, P_list, A_list, B_list, a_list, b_list)
        r_list = np.flip(r_traj_rev, axis=0)

        # z0 = -1.0 * np.linalg.inv(P_list[0]) @ r_list[0]
        z0 = np.zeros((self.dim, 1))
        z_list = self.z_traj_sim(z0, P_list, r_list, A_list, B_list, b_list)

        v_list = np.zeros((self.tsteps, self.dim, 1))
        for t in range(self.tsteps):
            zt = z_list[t]
            Pt = P_list[t]
            rt = r_list[t]
            Bt = B_list[t]
            bt = b_list[t]
            # print(self.z2v(zt, Pt, rt, Bt, bt))
            # input()
            v_list[t] = self.z2v(zt, Pt, rt, Bt, bt)

        return v_list
    
    def line_search(self, s0, u_traj, v_traj):
        s_traj = self.agent.traj_sim(s0, u_traj)
        opt_loss = self.loss(s_traj, u_traj)

        step_list = 1.0 * np.power(0.5, np.arange(10))
        # step_list = [1.0]
        opt_step = 0.0
        opt_s_traj = s_traj
        for step in step_list:
            temp_u_traj = u_traj + step * v_traj
            temp_s_traj = self.agent.traj_sim(s0, temp_u_traj)
            temp_loss = self.loss(temp_s_traj, temp_u_traj)
            if temp_loss < opt_loss:
                opt_loss = temp_loss
                opt_step = step
                opt_s_traj = temp_s_traj
                break

        return opt_step, opt_s_traj, opt_loss


def main():
    #################################################
    # Global parameters
    #################################################
    dt = 0.1
    dim = 3
    debug = False
    tsteps = 4

    agent = Agent(dt = dt, dim = dim)
    mu1 = np.array([1.269361904911189942e+00, -5.033011044967376062e-17, 1.088677623095679170e+00])
    mu1 = np.loadtxt("means.txt")
    vis_means = np.array([mu1])
    means = np.array([agent.eulerToSo3(mu1)])
    # covs = np.array([[8.975154274061541981e-02, -5.177742375229565953e-18, 3.474761085770630142e-01],
    #                  [-5.177742375229565953e-2, 1.000000000000000082e-01, -8.102804551067627008e-2],
    #                  [3.474761085770630142e-01, -8.102804551067627008e-18, 3.128979025482649057e+00]])
    covs = np.loadtxt("covs.txt")
    covs = np.array([covs])
    weights = np.ones(len(means)) / len(means)
    tgt = Distr(means, covs, weights)
    Q = np.eye(dim) * 1.
    Q[1, 1] *= 100.
    R = np.eye(dim) * 1.
    kernel = KernelSO3(theta = 16.644)
    ilqr = SO3_LQR(
        tsteps=tsteps,
        dt=dt,
        agent=agent,
        tgt=tgt,
        kernel=kernel,
        Q=Q, R=R,
        # info_w=15e-3,
        # explr_w=1e-1,
        info_w=1e-2,
        explr_w=5e-1,
        ctrl_w=0.,
        barr_w=0.25
        )
    
    # from optimize_params import SO3World, KernelObjective
    # from bootstrap import TSP
    # ko = KernelObjective(means, covs, weights, 50, 50)
    # input(ko.find_optimal())
    # world = SO3World(num_samples=tsteps, N=50)
    # world.load_distribution(means, covs, weights)
    # points = world.target_dist_samples()
    # tsp = TSP(points)
    # target_points = tsp.opt_2_algorithm()
    # s0 = target_points[0]
    # target_points = Rotation.from_matrix(target_points)
    # controls = np.zeros((tsteps, dim, 1))
    # current_rotation = Rotation.from_matrix(s0)
    # for i, desired_rotation in enumerate(target_points):
    #     controls[i] = np.expand_dims(agent.inv_step(desired_rotation.as_matrix(), current_rotation.as_matrix()), axis=1)
    #     current_rotation = desired_rotation
    # u_traj = controls.copy()
    
    s0 = vis_means[0].copy()
    # s0 = np.array([np.pi/4, 0., np.pi/6])
    # s0[0] -= 0.2
    # s0[2] -= 1.2
    s0 = agent.eulerToSo3(s0)
    u_traj = np.zeros((tsteps, dim, 1)) + 0.1
    u_traj = np.expand_dims(np.tile(np.array([0.05, 0.0, 0.5]), (tsteps//2, 1)), axis=2) / dt
    u_traj2 = np.expand_dims(np.tile(np.array([-0.05, 0.0, -0.5]), (tsteps//2 + tsteps%2, 1)), axis=2) / dt
    u_traj = np.concatenate((u_traj, u_traj2))

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
    fig, axes = plt.subplots(1, 3, figsize=(16,6), tight_layout=True)

    for iter in range(500):
        fig.suptitle(f'Iter: {iter}; Vis in Euclidean Space')

        v_traj = ilqr.get_v_traj(s0, u_traj)
        step, opt_s_traj, opt_loss = ilqr.line_search(s0, u_traj, v_traj)
        u_traj += step * v_traj

        print('loss: ', opt_loss)
        print('step: ', step)
        if (iter-2) % 10 == 0 or step == 0.0:
        # if iter > 0:
            straj_vis = agent.stToEuler(opt_s_traj)
            ax = axes[0]
            ax.cla()
            ax.set_aspect('auto')
            ax.set_title("Roll vs Pitch")
            ax.contourf(grids_r, grids_y, vis_rp, cmap='Reds')
            ax.plot(straj_vis[:,0], straj_vis[:,1], linestyle=" ", marker='o', color='k', markersize=3, alpha=0.5)

            ax = axes[1]
            ax.cla()
            ax.set_title("Roll vs Yaw")
            ax.set_aspect('auto')
            ax.contourf(grids_r, grids_y, vis_ry, cmap='Reds')
            ax.plot(straj_vis[:,0], straj_vis[:,2], linestyle=" ", marker='o', color='k', markersize=3, alpha=0.5)

            ax = axes[2]
            ax.cla()
            ax.set_title("Pitch vs Yaw")
            ax.set_aspect('auto')
            ax.contourf(grids_r, grids_y, vis_py, cmap='Reds')
            ax.plot(straj_vis[:,1], straj_vis[:,2], linestyle=" ", marker='o', color='k', markersize=3, alpha=0.5)

            plt.pause(0.01)
        
        if step <= 1e-4 and iter > 2:
            print("step is 0")
            break
            
    plt.show()
    plt.close()

    return opt_s_traj

    np.savetxt('s_traj.txt', opt_s_traj.reshape(-1, 3))

def ergodic_search(means, covs, weights, optimal_kernel, Q_diag=1., R_diag=1., info_w=1e-2, explr_w=5e-1, ctrl_w=0.0, dt=0.1, dim=3, tsteps=100, prev_traj=None):
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