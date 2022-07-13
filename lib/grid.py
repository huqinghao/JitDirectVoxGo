import os
import time
import functools
import numpy as np

# import jt
# import jt.nn as nn
# import jt.nn.functional as F
from jittor import init
import jittor as jt
import jittor.nn as nn
# import jt.nn.functional as F
from .jit_cuda import render_utils,total_variation,up_sample3d,grid_sampler
# from jt.utils.cpp_extension import load
#TODO
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# render_utils_cuda = load(
#         name='render_utils_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
#         verbose=True)

# total_variation_cuda = load(
#         name='total_variation_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/total_variation.cpp', 'cuda/total_variation_kernel.cu']],
#         verbose=True)


def create_grid(type, **kwargs):
    if type == 'DenseGrid':
        return DenseGrid(**kwargs)
    elif type == 'TensoRFGrid':
        return TensoRFGrid(**kwargs)
    else:
        raise NotImplementedError


''' Dense 3D grid
'''
class DenseGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, **kwargs):
        super(DenseGrid, self).__init__()
        self.channels = channels
        self._world_size = world_size.copy().stop_grad()
        # self.register_buffer('xyz_min', jt.Tensor(xyz_min))
        # self.register_buffer('xyz_max', jt.Tensor(xyz_max))
        self.xyz_min=jt.array(xyz_min.copy()).stop_grad()
        self.xyz_max=jt.array(xyz_max.copy()).stop_grad()
        # self.grid = (jt.zeros([1, channels, *world_size]))
        #TODO: jittor *Var
        self.grid =jt.zeros([1, channels, *world_size.tolist()])
    def execute(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        if xyz.numel()==0:
            return jt.array([])
        shape = xyz.shape[:-1]
        xyz_ = xyz.reshape(1,1,1,-1,3)
        ind_norm = ((xyz_ - self.xyz_min) / (self.xyz_max - self.xyz_min)).flip(-1) * 2 - 1
        # ind_norm = ((xyz_ - self.xyz_min) / (self.xyz_max - self.xyz_min)) * 2 - 1
        # TODO
        # nn.grid_sample too slow 
        #out = grid_sampler.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)
        # print("grid_min:{}".format(grid.min().item()))
        out = grid_sampler.grid_sample(self.grid, ind_norm, mode='bilinear', align_corners=True)

        # out = jt.randn((*shape,self.channels))
        # out= jt.Var(np.load("dense_grid.npy"))
        
        out = out.reshape(self.channels,-1).t().reshape(*shape,self.channels)
        if self.channels == 1:
            out = out.squeeze(-1)
        return out
    
    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            self.grid = jt.zeros([1, self.channels, *new_world_size])
        else:
            # self.grid = nn.interpolate(self.grid, size=tuple(new_world_size), \
            #     mode='trilinear', align_corners=True)
            self.grid = up_sample3d.interpolate(self.grid, size=tuple(new_world_size), \
                mode='trilinear', align_corners=True)
            #TODO: use new  optimizer  or the parameters will not 
            self.grid.requires_grad=True

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        #TODO: grad
        total_variation.total_variation_add_grad(
            self.grid, self.grid.grad, wx, wy, wz, dense_mode)

    def get_dense_grid(self):
        return self.grid

    @jt.no_grad()
    def __isub__(self, val):
        self.grid-= val
        self.grid.requires_grad=True
        return self

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self._world_size.tolist()}'


''' Vector-Matrix decomposited grid
See TensoRF: Tensorial Radiance Fields (https://arxiv.org/abs/2203.09517)
'''
class TensoRFGrid(nn.Module):
    def __init__(self, channels, world_size, xyz_min, xyz_max, config):
        super(TensoRFGrid, self).__init__()
        self.channels = channels
        self._world_size = world_size
        self.config = config
        self.xyz_min=jt.float32(xyz_min).stop_grad()
        self.xyz_max=jt.float32(xyz_max).stop_grad()
        # self.register_buffer('xyz_min', jt.Tensor(xyz_min))
        # self.register_buffer('xyz_max', jt.Tensor(xyz_max))
        X, Y, Z = world_size
        R = config['n_comp']
        Rxy = config.get('n_comp_xy', R)
        # self.xy_plane = (jt.randn([1, Rxy, X, Y]) * 0.1)
        # self.xz_plane = (jt.randn([1, R, X, Z]) * 0.1)
        # self.yz_plane = (jt.randn([1, R, Y, Z]) * 0.1)
        # self.x_vec = (jt.randn([1, R, X, 1]) * 0.1)
        # self.y_vec = (jt.randn([1, R, Y, 1]) * 0.1)
        # self.z_vec = (jt.randn([1, Rxy, Z, 1]) * 0.1)
        self.xy_plane = jt.randn([1, Rxy, X, Y]) * 0.1
        self.xz_plane = jt.randn([1, R, X, Z]) * 0.1
        self.yz_plane = jt.randn([1, R, Y, Z]) * 0.1
        self.x_vec = jt.randn([1, R, X, 1]) * 0.1
        self.y_vec =jt.randn([1, R, Y, 1]) * 0.1
        self.z_vec = jt.randn([1, Rxy, Z, 1]) * 0.1
        if self.channels > 1:
            # self.f_vec = (jt.ones([R+R+Rxy, channels]))
            self.f_vec = jt.ones([R+R+Rxy, channels])
            init.kaiming_uniform_(self.f_vec, a=np.sqrt(5))

    def execute(self, xyz):
        '''
        xyz: global coordinates to query
        '''
        shape = xyz.shape[:-1]
        xyz = xyz.reshape(1,1,-1,3)
        ind_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min) * 2 - 1
        ind_norm = jt.contrib.concat([ind_norm, jt.zeros_like(ind_norm[...,[0]])], dim=-1)
        if self.channels > 1:
            out = compute_tensorf_feat(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, self.f_vec, ind_norm)
            out = out.reshape(*shape,self.channels)
        else:
            out = compute_tensorf_val(
                    self.xy_plane, self.xz_plane, self.yz_plane,
                    self.x_vec, self.y_vec, self.z_vec, ind_norm)
            out = out.reshape(*shape)
        return out

    def scale_volume_grid(self, new_world_size):
        if self.channels == 0:
            return
        X, Y, Z = new_world_size
        # self.xy_plane = (F.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True))
        # self.xz_plane = (F.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True))
        # self.yz_plane = (F.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True))
        # self.x_vec = (F.interpolate(self.x_vec.data, size=[X,1], mode='bilinear', align_corners=True))
        # self.y_vec = (F.interpolate(self.y_vec.data, size=[Y,1], mode='bilinear', align_corners=True))
        # self.z_vec = (F.interpolate(self.z_vec.data, size=[Z,1], mode='bilinear', align_corners=True))
        self.xy_plane = nn.interpolate(self.xy_plane.data, size=[X,Y], mode='bilinear', align_corners=True)
        self.xz_plane =nn.interpolate(self.xz_plane.data, size=[X,Z], mode='bilinear', align_corners=True)
        self.yz_plane = nn.interpolate(self.yz_plane.data, size=[Y,Z], mode='bilinear', align_corners=True)
        self.x_vec = nn.interpolate(self.x_vec.data, size=[X,1], mode='bilinear', align_corners=True)
        self.y_vec = nn.interpolate(self.y_vec.data, size=[Y,1], mode='bilinear', align_corners=True)
        self.z_vec = nn.interpolate(self.z_vec.data, size=[Z,1], mode='bilinear', align_corners=True)

    def total_variation_add_grad(self, wx, wy, wz, dense_mode):
        '''Add gradients by total variation loss in-place'''
        loss = wx * nn.smooth_l1_loss(self.xy_plane[:,:,1:], self.xy_plane[:,:,:-1], reduction='sum') +\
               wy * nn.smooth_l1_loss(self.xy_plane[:,:,:,1:], self.xy_plane[:,:,:,:-1], reduction='sum') +\
               wx * nn.smooth_l1_loss(self.xz_plane[:,:,1:], self.xz_plane[:,:,:-1], reduction='sum') +\
               wz * nn.smooth_l1_loss(self.xz_plane[:,:,:,1:], self.xz_plane[:,:,:,:-1], reduction='sum') +\
               wy * nn.smooth_l1_loss(self.yz_plane[:,:,1:], self.yz_plane[:,:,:-1], reduction='sum') +\
               wz * nn.smooth_l1_loss(self.yz_plane[:,:,:,1:], self.yz_plane[:,:,:,:-1], reduction='sum') +\
               wx * nn.smooth_l1_loss(self.x_vec[:,:,1:], self.x_vec[:,:,:-1], reduction='sum') +\
               wy * nn.smooth_l1_loss(self.y_vec[:,:,1:], self.y_vec[:,:,:-1], reduction='sum') +\
               wz * nn.smooth_l1_loss(self.z_vec[:,:,1:], self.z_vec[:,:,:-1], reduction='sum')
        loss /= 6
        loss.backward()

    def get_dense_grid(self):
        if self.channels > 1:
            feat = jt.contrib.concat([
                jt.einsum('rxy,rz->rxyz', self.xy_plane[0], self.z_vec[0,:,:,0]),
                jt.einsum('rxz,ry->rxyz', self.xz_plane[0], self.y_vec[0,:,:,0]),
                jt.einsum('ryz,rx->rxyz', self.yz_plane[0], self.x_vec[0,:,:,0]),
            ])
            grid = jt.einsum('rxyz,rc->cxyz', feat, self.f_vec)[None]
        else:
            grid = jt.einsum('rxy,rz->xyz', self.xy_plane[0], self.z_vec[0,:,:,0]) + \
                   jt.einsum('rxz,ry->xyz', self.xz_plane[0], self.y_vec[0,:,:,0]) + \
                   jt.einsum('ryz,rx->xyz', self.yz_plane[0], self.x_vec[0,:,:,0])
            grid = grid[None,None]
        return grid

    def extra_repr(self):
        return f'channels={self.channels}, world_size={self._world_size.tolist()}, n_comp={self.config["n_comp"]}'

def compute_tensorf_feat(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, f_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = nn.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = nn.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = nn.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = nn.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = nn.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = nn.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = jt.contrib.concat([
        xy_feat * z_feat,
        xz_feat * y_feat,
        yz_feat * x_feat,
    ], dim=-1)
    feat = jt.matmul(feat, f_vec)
    return feat

def compute_tensorf_val(xy_plane, xz_plane, yz_plane, x_vec, y_vec, z_vec, ind_norm):
    # Interp feature (feat shape: [n_pts, n_comp])
    xy_feat = nn.grid_sample(xy_plane, ind_norm[:,:,:,[1,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    xz_feat = nn.grid_sample(xz_plane, ind_norm[:,:,:,[2,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    yz_feat = nn.grid_sample(yz_plane, ind_norm[:,:,:,[2,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    x_feat = nn.grid_sample(x_vec, ind_norm[:,:,:,[3,0]], mode='bilinear', align_corners=True).flatten(0,2).T
    y_feat = nn.grid_sample(y_vec, ind_norm[:,:,:,[3,1]], mode='bilinear', align_corners=True).flatten(0,2).T
    z_feat = nn.grid_sample(z_vec, ind_norm[:,:,:,[3,2]], mode='bilinear', align_corners=True).flatten(0,2).T
    # Aggregate components
    feat = (xy_feat * z_feat).sum(-1) + (xz_feat * y_feat).sum(-1) + (yz_feat * x_feat).sum(-1)
    return feat


''' Mask grid
It supports query for the known free space and unknown space.
'''
class MaskGrid(nn.Module):
    def __init__(self, path=None, mask_cache_thres=None, mask=None, xyz_min=None, xyz_max=None):
        super(MaskGrid, self).__init__()
        if path is not None:
            st = jt.load(path)
            self.mask_cache_thres = mask_cache_thres
            density = nn.max_pool3d(jt.array(st['model_state_dict']['density.grid']), kernel_size=3, stride=1, padding=1)
            alpha = 1 - jt.exp(-nn.softplus(density + st['model_state_dict']['act_shift']) * st['model_kwargs']['voxel_size_ratio'])
            mask = (alpha >= self.mask_cache_thres).squeeze(0).squeeze(0)
            mask = mask.bool().stop_grad()
            xyz_min = jt.float32(st['model_kwargs']['xyz_min']).stop_grad()
            xyz_max = jt.float32(st['model_kwargs']['xyz_max']).stop_grad()
        else:
            mask = mask.bool().stop_grad()
            xyz_min = jt.float32(xyz_min).stop_grad()
            xyz_max = jt.float32(xyz_max).stop_grad()

        self.mask=mask.stop_grad()
        xyz_len = xyz_max - xyz_min
        self.xyz2ijk_scale= ((jt.float32(list(mask.shape)) - 1) / xyz_len).stop_grad()
        self.xyz2ijk_shift= (-xyz_min * self.xyz2ijk_scale).stop_grad()

    @jt.no_grad()
    def execute(self, xyz):
        '''Skip know freespace
        @xyz:   [..., 3] the xyz in global coordinate.
        '''
        shape = xyz.shape[:-1]
        _xyz = xyz.reshape(-1, 3)
        #TODO:not implemented yet
        mask = render_utils.maskcache_lookup(self.mask, _xyz, self.xyz2ijk_scale, self.xyz2ijk_shift)
        # mask=jt.Var(np.load("mask.npy"))
        mask = mask.reshape(shape)
        return mask

    def extra_repr(self):
        return f'mask.shape=list(self.mask.shape)'

