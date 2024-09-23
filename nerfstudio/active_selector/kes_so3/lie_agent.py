import matplotlib.pyplot as plt 
from tqdm import tqdm 
import modern_robotics as mr
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import multivariate_normal as mvn

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
