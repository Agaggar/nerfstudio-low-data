"""
Implementing entropy based on weights, as per Lee et al. https://arxiv.org/pdf/2209.08409
"""

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt

def compute_entropy(weights: Tensor):
    # if bool(torch.any(torch.sum(weights, dim=-2, keepdim=True) == 0)):
    #     return torch.zeros_like(torch.sum(weights, dim=-2, keepdim=True))
    p = weights # / torch.sum(weights, dim=-2, keepdim=True)
    p /= torch.max(p)
    p = torch.nan_to_num(p)
    # p[torch.argwhere(torch.sum(weights, dim=-2, keepdim=True) == 0)] = 0.
    # p /= torch.sum(p, dim=-2, keepdim=True)
    if bool(torch.any(p == 0)):
        return -10*torch.sum(p * torch.log2(p + 1e-9), dim=-2)
    return -10*torch.sum(p * torch.log2(p), dim=-2)

def compute_entropy_ngp(weights: Tensor):
    # if bool(torch.any(torch.sum(weights, dim=-2, keepdim=True) == 0)):
    #     return torch.zeros_like(torch.sum(weights, dim=-2, keepdim=True))
    p = weights # / torch.sum(weights, dim=-2, keepdim=True)
    p /= torch.max(p)
    p = torch.nan_to_num(p)
    if bool(torch.any(p == 0)):
        return -10*torch.sum(p * torch.log2(p + 1e-9), dim=1)
    return -10*torch.sum(p * torch.log2(p), dim=1)

def plot_opacity_weight(opacity, weights, entropy_val=None):
    plt.plot(opacity.cpu(), label='opacity')
    plt.plot(weights.cpu(), label='weights')
    plt.legend()
    plt.title('Entropy val ' + str(torch.sum(entropy_val).item()))
    plt.show()
    plt.close()

def lowpass_filter(signal, alpha=0.1):
    """
    Applies a simple lowpass filter to the input signal.
    
    Parameters:
    - signal: Input signal (1D NumPy array)
    - alpha: Smoothing factor (between 0 and 1)
    
    Returns:
    - filtered_signal: The filtered signal (1D NumPy array)
    """
    filtered_signal = torch.zeros_like(signal)
    filtered_signal[0] = signal[0]  # Initialize with the first value

    # Apply the lowpass filter
    for i in range(1, len(signal)):
        filtered_signal[i] = alpha * signal[i] + (1 - alpha) * filtered_signal[i-1]
    
    return filtered_signal