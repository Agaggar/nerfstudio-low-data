#####################################################
# SO3 Kernel Ergodic iLQR Control - Cost Functions
# Author: Ayush Gaggar, adapted from https://murpheylab.github.io/pdfs/2016RSSFaMu.pdf
# Date: 04-25-2024
#####################################################

import numpy as np
import modern_robotics as mr

def quadratic_cost(g1, g2, M) -> float:
    '''
    quadatic cost equation (eq 29)
    params:
    - g1: SO(3), 3x3 matrix
    - g2: SO(3), 3x3 matrix
    - M: 3x3 matrix

    returns:
    float value of cost
    '''
    diff_mat = (mr.RotInv(g2) @ g1)
    diff_vec = mr.so3ToVec(mr.MatrixLog3(diff_mat))
    diff_vec = np.expand_dims(diff_vec, axis=1)
    return (diff_vec.T @ M @ diff_vec).flatten()[0]

def d_exp_inv_mat(g1, g2, M) -> np.ndarray:
    '''
    derivative of exp inv matrix (eq 31b)
    params:
    - g1: SO(3), 3x3 matrix
    - g2: SO(3), 3x3 matrix
    - M: 3x3 matrix

    returns:
    3x3 matrix of d_exp_inv
    '''
    gd_inv_g = (mr.RotInv(g2) @ g1)
    neg_diff_so3 = -1.0 * mr.MatrixLog3(gd_inv_g)
    neg_diff_vec = mr.so3ToVec(neg_diff_so3)
    # neg_diff_vec = np.expand_dims(neg_diff_vec, axis=1)
    norm_vec = np.linalg.norm(neg_diff_vec)
    if norm_vec < 1e-6:
        return np.eye(3) - 0.5 * neg_diff_so3
    coef1 = (0.5 * norm_vec * np.sin(norm_vec) + np.cos(norm_vec) - 1) / (norm_vec**2 * (np.cos(norm_vec) - 1))
    dexp_inv = np.eye(3) - 0.5 * neg_diff_so3 + coef1 * (neg_diff_so3 @ neg_diff_so3)
    return dexp_inv

def d1_quad_cost(g1, g2, M) -> np.ndarray:
    '''
    D1 quadatic cost equation, i.e., derivative wrt state (eq 30)
    params:
    - g1: SO(3), 3x3 matrix
    - g2: SO(3), 3x3 matrix
    - M: 3x3 matrix

    returns:
    3x3 matrix of Dgf
    '''
    diff_mat = (mr.RotInv(g2) @ g1)
    diff_vec = mr.so3ToVec(diff_mat)
    diff_vec = np.expand_dims(diff_vec, axis=1)
    return d_exp_inv_mat(g1, g2, M).T @ M @ diff_vec

def d_exp_mat(g1, g2, M) -> np.ndarray:
    '''
    derivative of exp inv matrix (eq 31b)
    params:
    - g1: SO(3), 3x3 matrix
    - g2: SO(3), 3x3 matrix
    - M: 3x3 matrix

    returns:
    3x3 matrix of d_exp
    '''
    gd_inv_g = (mr.RotInv(g2) @ g1)
    neg_diff_so3 = -1.0 * mr.MatrixLog3(gd_inv_g)
    neg_diff_vec = mr.so3ToVec(neg_diff_so3)
    # neg_diff_vec = np.expand_dims(neg_diff_vec, axis=1)
    norm_vec = np.linalg.norm(neg_diff_vec)
    if norm_vec < 1e-6:
        return np.eye(3)
    coef1 = (1 - np.cos(norm_vec)) / (norm_vec**2)
    coef2 = (norm_vec - np.sin(norm_vec)) / (norm_vec**3)
    dexp = np.eye(3) + coef1 * neg_diff_so3 + coef2 * (neg_diff_so3 @ neg_diff_so3)
    return dexp
