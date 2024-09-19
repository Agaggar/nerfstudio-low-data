import matplotlib.pyplot as plt 
import numpy as np

from .so3_cost import quadratic_cost as dist
from .optimize_params import SO3World
from .ilqr_kes import GMM_Euclidean

class TSP:
    def __init__(self, points, dim=3) -> None:
        self.points: np.ndarray = points
        self.dim = dim

    def set_points(self, new_points: np.ndarray):
        self.points = new_points.copy()

    def total_distance(self, points):
        total = 0.0
        for i in range(len(points)-1):
            total += dist(points[i], points[i+1], M=np.eye(self.dim))
        return total
    
    def swap_2(self, one, two):
        points = self.points.copy()
        points[one] = self.points[two]
        points[two] = self.points[one]
        return points
    
    def opt_2_algorithm(self, improvement_threshold=1e-3):
        improvement_factor = 1.
        best_distance = self.total_distance(self.points)
        # print("current", best_distance)
        while improvement_factor > improvement_threshold:
            distance_to_beat = best_distance
            for swap_first in range(1, len(self.points)):
                for swap_second in range(swap_first+1, len(self.points)):
                    new_points = self.swap_2(swap_first, swap_second)
                    new_distance = self.total_distance(new_points)
                    if new_distance < best_distance:
                        best_distance = new_distance
                        # print(best_distance)
                        self.set_points(new_points)
            improvement_factor = 1 - best_distance / distance_to_beat
        
        return self.points
    

def main():
    world = SO3World(num_samples=5, N=50)
    means = world.agent.eulerToSo3(np.loadtxt("means.txt").reshape(-1, 3))
    covs = np.loadtxt("covs.txt").reshape(-1, 3, 3)
    weights = np.loadtxt("weights.txt").reshape(-1, 1)
    vis_means = world.agent.stToEuler(means)
    world.load_distribution(means, covs, weights)
    points = world.target_dist_samples()
    np.savetxt("original_points.txt", points.reshape(-1, 3))
    tsp = TSP(points)
    np.savetxt("tsp_opt.txt", tsp.opt_2_algorithm().reshape(-1, 3))

def plot():
    #TODO: overlay with gaussian to make sure the samples are actually doing something reasonable
    world = SO3World(num_samples=5, N=50)
    tsp_points = world.agent.stToEuler(np.loadtxt("tsp_opt.txt").reshape(-1, 3, 3))
    original_points = world.agent.stToEuler(np.loadtxt("original_points.txt").reshape(-1, 3, 3))

    means = world.agent.eulerToSo3(np.loadtxt("means.txt").reshape(-1, 3))
    covs = np.loadtxt("covs.txt").reshape(-1, 3, 3)
    weights = np.loadtxt("weights.txt").reshape(-1, 1)
    vis_means = world.agent.stToEuler(means)
    def vis():
        grids_r, grids_p, grids_y = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids = np.array([grids_r.ravel(), grids_p.ravel(), grids_y.ravel()]).T 
        
        # grids_r, grids_p = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(0, np.pi/2, 100)])
        grids_r, grids_p = np.meshgrid(*[np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)])
        grids_rp = np.array([grids_r.ravel(), grids_p.ravel()]).T
        vis_covs = np.empty([len(means), 2, 2])
        for i in range(len(means)):
            vis_covs[i, 0, 0] = covs[i, 0, 0]
            vis_covs[i, 0, 1] = covs[i, 0, 1]
            vis_covs[i, 1, 0] = covs[i, 1, 0]
            vis_covs[i, 1, 1] = covs[i, 1, 1]
        
        # vis_rp = GMM_Euclidean(
        #     np.array([mu1[0], mu1[1]]),
        #     vis_covs, np.ones(len(means))/len(means)).pdf(
        #                                 grids_rp).reshape((100, 100))
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
        # vis_ry = GMM_Euclidean(
        #     np.array([mu1[0], mu1[2]]),
        #     vis_covs, np.ones(len(means))/len(means)).pdf(
        #                                 grids_ry).reshape((100, 100))
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
        # vis_py = GMM_Euclidean(
        #     np.array([mu1[1], mu1[2]]),
        #     vis_covs, np.ones(len(means))/len(means)).pdf(
        #                                 grids_py).reshape((100, 100))
        vis_py = GMM_Euclidean(
            np.hstack(
                (np.expand_dims(vis_means[:, 1], axis=1),
                np.expand_dims(vis_means[:, 2], axis=1))),
            vis_covs, np.ones(len(means))/len(means)).pdf(
                                        grids_py).reshape((100, 100))

        return vis_rp, vis_ry, vis_py, grids_r, grids_p, grids_y
    
    vis_rp, vis_ry, vis_py, grids_r, grids_p, grids_y = vis()
    fig, ax = plt.subplots(1, 3)
    axis = [0, 1]
    ax[axis[0]].contourf(grids_r, grids_y, vis_rp, cmap='Reds')
    ax[axis[0]].scatter(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # ax[axis[0]].plot(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # ax[axis[0]].plot(original_points[:, axis[0]], original_points[:, axis[1]], linestyle="--", c="k")
    ax[axis[0]].set_title("Roll vs Pitch")

    axis = [0, 2]
    ax[1].contourf(grids_r, grids_y, vis_ry, cmap='Reds')
    ax[1].scatter(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # ax[1].plot(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # ax[1].plot(original_points[:, axis[0]], original_points[:, axis[1]], linestyle="--", c="k")
    ax[1].set_title("Roll vs Yaw")

    axis = [1, 2]
    ax[2].contourf(grids_r, grids_y, vis_py, cmap='Reds')
    ax[axis[1]].scatter(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # ax[axis[1]].plot(tsp_points[:, axis[0]], tsp_points[:, axis[1]], c="g")
    # ax[axis[1]].plot(original_points[:, axis[0]], original_points[:, axis[1]], linestyle="--", c="k")
    ax[axis[1]].set_title("Pitch vs Yaw")
    
    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
    plot()