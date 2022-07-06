import os, math
import numpy as np
import scipy.signal
from typing import List, Optional

import jittor as jt
from jittor import init
from jittor import nn
# from jt import Tensor

# import jt
# import jt.nn as nn
# import jt.nn.functional as F

from .masked_adam import MaskedAdam

def squeeze(input):
    in_shape = input.shape
    start_dim = 0
    end_dim = len(in_shape)
    out_shape = []
    for i in range(start_dim, end_dim, 1):
        if in_shape[i]!=1:
            out_shape.append(in_shape[i])
    return input.reshape(out_shape)

def randint(high, shape=(1,), dtype="int32"):
    ''' samples random integers from a uniform distribution on the interval [low, high).

    :param high: One above the highest integer to be drawn from the distribution.
    :type high: int
        
    :param shape: shape of the output size, defaults to (1,).
    :type shape: tuple, optional
        
    :param dtype: data type of the output, defaults to "int32".
    :type dtype: str, optional

    Example:
        
        >>> jt.randint(3, shape=(3, 3))
        jt.Var([[2 0 2]
         [2 1 2]
         [2 0 1]], dtype=int32)
        >>> jt.randint(1, 3, shape=(3, 3))
        jt.Var([[2 2 2]
         [1 1 2]
         [1 1 1]], dtype=int32)
    '''
    low=0
    v = (jt.random(shape) * (high - low) + low).clamp(low, high-0.5)
    v = jt.floor_int(v)
    return v.astype(dtype)
def filter_parameters(parameters):
    '''
    only Vars with require_grad=True are returned
    '''
    params=[]
    for p in parameters:
        if p.requires_grad:
            params.append(p)
    return params

def log10(x):
    return jt.log(x)/math.log(10.0)
''' Misc
'''
mse2psnr = lambda x : -10. * log10(x)
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def create_optimizer_or_freeze_model(model, cfg_train, global_step):
    decay_steps = cfg_train.lrate_decay * 1000
    decay_factor = 0.1 ** (global_step/decay_steps)
    
    param_group = []
    for k in cfg_train.keys():
        if not k.startswith('lrate_'):
            continue
        k = k[len('lrate_'):]

        if not hasattr(model, k):
            continue

        param = getattr(model, k)
        if param is None:
            print(f'create_optimizer_or_freeze_model: param {k} not exist')
            continue

        lr = getattr(cfg_train, f'lrate_{k}') * decay_factor
        if lr > 0:
            print(f'create_optimizer_or_freeze_model: param {k} lr {lr}')
            if isinstance(param, nn.Module):
                #TODO: nn.Module.parameters() return all vars including those Vars with requires_grad=False
                #To match with pytorch, filter out those buffer vars
                param = filter_parameters(param.parameters())
            param_group.append({'params': param, 'lr': lr, 'skip_zero_grad': (k in cfg_train.skip_zero_grad_fields)})
        else:
            print(f'create_optimizer_or_freeze_model: param {k} freeze')
            param.requires_grad = False
    return MaskedAdam(param_group,lr=lr)


''' Checkpoint utils
'''
def load_checkpoint(model, optimizer, ckpt_path, no_reload_optimizer):
    ckpt = jt.load(ckpt_path)
    start = ckpt['global_step']
    model.load_state_dict(ckpt['model_state_dict'])
    if not no_reload_optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return model, optimizer, start


def load_model(model_class, ckpt_path):
    ckpt = jt.load(ckpt_path)
    model = model_class(**ckpt['model_kwargs'])
    model.load_state_dict(ckpt['model_state_dict'])
    return model


''' Evaluation metrics (ssim, lpips)
'''
def rgb_ssim(img0, img1, max_val,
             filter_size=11,
             filter_sigma=1.5,
             k1=0.01,
             k2=0.03,
             return_map=False):
    # Modified from https://github.com/google/mipnerf/blob/16e73dfdb52044dcceb47cda5243a686391a6e0f/internal/math.py#L58
    assert len(img0.shape) == 3
    assert img0.shape[-1] == 3
    assert img0.shape == img1.shape

    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((np.arange(filter_size) - hw + shift) / filter_sigma)**2
    filt = np.exp(-0.5 * f_i)
    filt /= np.sum(filt)

    # Blur in x and y (faster than the 2D convolution).
    def convolve2d(z, f):
        return scipy.signal.convolve2d(z, f, mode='valid')

    filt_fn = lambda z: np.stack([
        convolve2d(convolve2d(z[...,i], filt[:, None]), filt[None, :])
        for i in range(z.shape[-1])], -1)
    mu0 = filt_fn(img0)
    mu1 = filt_fn(img1)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filt_fn(img0**2) - mu00
    sigma11 = filt_fn(img1**2) - mu11
    sigma01 = filt_fn(img0 * img1) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    sigma00 = np.maximum(0., sigma00)
    sigma11 = np.maximum(0., sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(
        np.sqrt(sigma00 * sigma11), np.abs(sigma01))
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim = np.mean(ssim_map)
    return ssim_map if return_map else ssim


__LPIPS__ = {}
def init_lpips(net_name, device):
    assert net_name in ['alex', 'vgg']
    import lpips
    print(f'init_lpips: lpips_{net_name}')
    return lpips.LPIPS(net=net_name, version='0.1').eval()
    # return lpips.LPIPS(net=net_name, version='0.1').eval().to(device)


def rgb_lpips(np_gt, np_im, net_name, device):
    if net_name not in __LPIPS__:
        __LPIPS__[net_name] = init_lpips(net_name, device)
    # gt = jt.array(np_gt).permute([2, 0, 1]).contiguous().to(device)
    # im = jt.array(np_im).permute([2, 0, 1]).contiguous().to(device)
    gt = jt.array(np_gt).permute([2, 0, 1])
    im = jt.array(np_im).permute([2, 0, 1])
    return __LPIPS__[net_name](gt, im, normalize=True).item()

