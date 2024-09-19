import numpy as np 
import matplotlib.pyplot as plt 
from tqdm import tqdm 

from .ilqr_kes import Agent, KernelSO3, Distr

################## World Setup ###########################
class SO3World:
    def __init__(self, num_samples=200, N=100, seed=0, dt=0.1):
        pass
        self.seed = seed
        self.num_samples = num_samples
        self.N = N
        self.rng = np.random.default_rng(seed=self.seed)
        self.agent = Agent(dt = dt)

    def load_distribution(self, means, covs, weights):
        self.means = means
        self.covs = covs
        self.weights = weights
        self.tgt = Distr(self.means, self.covs, self.weights)

    def target_dist_samples(self):
        grids_r, grids_p, grids_y = np.meshgrid(*[np.linspace(-np.pi/2, np.pi/2, self.N), 
                                                  np.linspace(0., 0., self.N), 
                                                  np.linspace(-np.pi, np.pi, self.N)])
        grids = np.array([grids_r.ravel(), grids_p.ravel(), grids_y.ravel()]).T 
        grids = self.agent.eulerToSo3(grids)
        pdf_grids = np.zeros(len(grids))
        for i, st in tqdm(enumerate(grids), total=len(grids)):
            pdf_grids[i] = self.tgt.pdf(st)
        pdf_grids /= np.sum(pdf_grids)
        # pdf_grids[pdf_grids < (np.max(pdf_grids) * 0.98)] = 0.0
        # pdf_grids /= np.sum(pdf_grids)
        samples = self.rng.choice(grids, size=self.num_samples, replace=False, p=pdf_grids)
        return samples

class KernelObjective:
    def __init__(self, means, covs, weights, num_samples=100, N=50, seed=0, dt=0.1):
        self.world = SO3World(num_samples, N, seed, dt)
        self.world.load_distribution(means, covs, weights)
        self.samples = self.world.target_dist_samples()
        self.dinfo_gain = self.d_info_gain()

    def d_info_gain(self):
        dpdf_vals = np.zeros((len(self.samples), 3, 1))
        for i, st in enumerate(self.samples):
            dpdf_vals[i] = self.world.tgt.dpdf(st)
        
        return -2.0 * np.mean(dpdf_vals, axis=0)
    
    def obj(self, kernel_param, info_gain=1.0, explr_gain=1.0, total=None):
        kernel = KernelSO3(theta=kernel_param)

        explr_loss = np.zeros((3, 1))
        if total is None:
            for st1 in self.samples.copy():
                for st2 in self.samples.copy():
                    explr_loss += kernel.d1_eval(st1, st2)
        else:
            for st1 in self.samples.copy()[:total]:
                for st2 in self.samples.copy()[:total]:
                    explr_loss += kernel.d1_eval(st1, st2)
        explr_loss /= len(self.samples) * len(self.samples)

        return np.sum(self.dinfo_gain * info_gain + explr_loss * explr_gain)
    
    def find_optimal(self):
        param_opts = np.power(10.0, np.arange(-5, 5))
        obj_val = np.zeros(len(param_opts))
        for i, param in tqdm(enumerate(param_opts), total=len(param_opts)):
            obj_val[i] = self.obj(param)
        print("coarse", param_opts[np.argmin(obj_val)])
        param_opts = np.linspace(1, 100, 150) * param_opts[np.argmin(obj_val)]
        obj_val = np.zeros(len(param_opts))
        for i, param in tqdm(enumerate(param_opts), total=len(param_opts)):
            obj_val[i] = self.obj(param)
        return param_opts[np.argmin(obj_val)]

    def tune_ws(self, kernel_val, num_views=5):
        info_w = np.linspace(0., 10., 100)
        explr_w = np.linspace(0., 10., 100)
        obj_val = np.zeros((len(info_w), len(explr_w)))
        for i, i_w in tqdm(enumerate(info_w), total=len(obj_val)):
            for j, e_w in enumerate(explr_w):
                obj_val[i, j] = self.obj(kernel_val, i_w, e_w, total=num_views)
        min_ind = np.unravel_index(obj_val.argmin(), obj_val.shape)
        return info_w[min_ind[0]], explr_w[min_ind[1]]
