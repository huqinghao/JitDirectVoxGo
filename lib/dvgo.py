import os
import time
import functools
import numpy as np

import jittor as jt
from jittor import init
from jittor import nn
# import jt
# import jt.nn as nn
# import jt.nn.functional as F

#TODO
# from torch_scatter import segment_coo
from jittor import scatter

from jittor import Function
from . import grid

#TODO: utils
import time
from lib.jit_cuda import render_utils
# from jt.utils.cpp_extension import load
# parent_dir = os.path.dirname(os.path.abspath(__file__))
# render_utils_cuda = load(
#         name='render_utils_cuda',
#         sources=[
#             os.path.join(parent_dir, path)
#             for path in ['cuda/render_utils.cpp', 'cuda/render_utils_kernel.cu']],
#         verbose=True)
#TODO: check the value
# raw2alpha = lambda raw, shift,interval: 1.-jt.pow(1+jt.exp(raw+shift),-interval)

# alpha2weights = lambda raw, shift,interval: 1.-jt.pow(1+jt.exp(raw+shift),-interval)

'''Model'''
class DirectVoxGO(jt.nn.Module):
    def __init__(self, xyz_min, xyz_max,
                 num_voxels=0, num_voxels_base=0,
                 alpha_init=None,
                 mask_cache_path=None, mask_cache_thres=1e-3, mask_cache_world_size=None,
                 fast_color_thres=0,
                 density_type='DenseGrid', k0_type='DenseGrid',
                 density_config={}, k0_config={},
                 rgbnet_dim=0, rgbnet_direct=False, rgbnet_full_implicit=False,
                 rgbnet_depth=3, rgbnet_width=128,
                 viewbase_pe=4,
                 **kwargs):
        super(DirectVoxGO, self).__init__()
        self.xyz_min=jt.float32(xyz_min).stop_grad()
        self.xyz_max=jt.float32(xyz_max).stop_grad()
        self.fast_color_thres = fast_color_thres

        # determine based grid resolution
        
        self.num_voxels_base = num_voxels_base
        #TODO: make it private
        self._voxel_size_base = ((self.xyz_max - self.xyz_min).prod() / self.num_voxels_base).pow(1/3)

        # determine the density bias shift
        self.alpha_init = alpha_init
        #self.register_buffer('act_shift', jt.FloatTensor([np.log(1/(1-alpha_init) - 1)]))
        self.act_shift=jt.float32([np.log(1/(1-alpha_init) - 1)]).stop_grad()

        print('dvgo: set density bias shift to', self.act_shift)

        # determine init grid resolution
        self._set_grid_resolution(num_voxels)

        # init density voxel grid
        self.density_type = density_type
        self.density_config = density_config
        self.density = grid.create_grid(
                density_type, channels=1, world_size=self._world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.density_config)

        # init color representation
        self.rgbnet_kwargs = {
            'rgbnet_dim': rgbnet_dim, 'rgbnet_direct': rgbnet_direct,
            'rgbnet_full_implicit': rgbnet_full_implicit,
            'rgbnet_depth': rgbnet_depth, 'rgbnet_width': rgbnet_width,
            'viewbase_pe': viewbase_pe,
        }
        self.k0_type = k0_type
        self.k0_config = k0_config
        self.rgbnet_full_implicit = rgbnet_full_implicit
        if rgbnet_dim <= 0:
            # color voxel grid  (coarse stage)
            self.k0_dim = 3
            self.k0 = grid.create_grid(
                k0_type, channels=self.k0_dim, world_size=self._world_size,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                config=self.k0_config)
            self.rgbnet = None
        else:
            # feature voxel grid + shallow MLP  (fine stage)
            if self.rgbnet_full_implicit:
                self.k0_dim = 0
            else:
                self.k0_dim = rgbnet_dim
            self.k0 = grid.create_grid(
                    k0_type, channels=self.k0_dim, world_size=self._world_size,
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max,
                    config=self.k0_config)
            self.rgbnet_direct = rgbnet_direct

            self.viewfreq= jt.float32([(2 ** i) for i in range(viewbase_pe)]).stop_grad()
            dim0 = int(3+3*viewbase_pe*2)
            if self.rgbnet_full_implicit:
                pass
            elif rgbnet_direct:
                dim0 += self.k0_dim
            else:
                dim0 += self.k0_dim-3
            self.rgbnet = nn.Sequential(
                nn.Linear(dim0, rgbnet_width), nn.ReLU(),
                *[
                    nn.Sequential(nn.Linear(rgbnet_width, rgbnet_width), nn.ReLU())
                    for _ in range(rgbnet_depth-2)
                ],
                nn.Linear(rgbnet_width, 3),
            )
            init.constant_(self.rgbnet[(- 1)].bias, value=0)
            # nn.init.constant_(self.rgbnet[-1].bias, 0)
            print('dvgo: feature voxel grid', self.k0)
            print('dvgo: mlp', self.rgbnet)

        # Using the coarse geometry if provided (used to determine known free space and unknown space)
        # Re-implement as occupancy grid (2021/1/31)
        self.mask_cache_path = mask_cache_path
        self.mask_cache_thres = mask_cache_thres
        if mask_cache_world_size is None:
            mask_cache_world_size = self._world_size
        else:
            mask_cache_world_size = jt.array(mask_cache_world_size,dtype='int32')
        if mask_cache_path is not None and mask_cache_path:
            # mask_cache = grid.MaskGrid(
            #         path=mask_cache_path,
            #         mask_cache_thres=mask_cache_thres).to(self.xyz_min.device)
            mask_cache = grid.MaskGrid(
                    path=mask_cache_path,
                    mask_cache_thres=mask_cache_thres)
            self_grid_xyz = jt.stack(jt.meshgrid(
                jt.linspace(self.xyz_min[0], self.xyz_max[0], mask_cache_world_size[0].item()),
                jt.linspace(self.xyz_min[1], self.xyz_max[1], mask_cache_world_size[1].item()),
                jt.linspace(self.xyz_min[2], self.xyz_max[2], mask_cache_world_size[2].item()),
            ), -1)
            mask = mask_cache(self_grid_xyz)
        else:
            #TODO: list(Var)==>list not supported
            # mask = jt.ones(list(mask_cache_world_size), dtype=jt.bool)
            mask = jt.ones([ind.item() for ind in mask_cache_world_size], dtype=jt.bool)
            # mask = jt.ones(mask_cache_world_size, dtype=jt.bool)
        self.mask_cache = grid.MaskGrid(
                path=None, mask=mask,
                xyz_min=self.xyz_min, xyz_max=self.xyz_max)

    def _set_grid_resolution(self, num_voxels):
        # Determine grid resolution
        self.num_voxels = num_voxels
        #TODO: make it private
        self._voxel_size = ((self.xyz_max - self.xyz_min).prod() / num_voxels).pow(1/3).stop_grad()
        ##TODO: make it private
        self._world_size = ((self.xyz_max - self.xyz_min) / self._voxel_size).long().stop_grad()
        #TODO: make it private
        self._voxel_size_ratio = (self._voxel_size / self._voxel_size_base).stop_grad()
        print('dvgo: voxel_size      ', self._voxel_size)
        print('dvgo: world_size      ', self._world_size)
        print('dvgo: voxel_size_base ', self._voxel_size_base)
        print('dvgo: voxel_size_ratio', self._voxel_size_ratio)

    def get_kwargs(self):
        return {
            # 'xyz_min': self.xyz_min.cpu().numpy(),
            # 'xyz_max': self.xyz_max.cpu().numpy(),
            'xyz_min': self.xyz_min.numpy(),
            'xyz_max': self.xyz_max.numpy(),
            'num_voxels': self.num_voxels,
            'num_voxels_base': self.num_voxels_base,
            'alpha_init': self.alpha_init,
            'voxel_size_ratio': self._voxel_size_ratio,
            'mask_cache_path': self.mask_cache_path,
            'mask_cache_thres': self.mask_cache_thres,
            'mask_cache_world_size': list(self.mask_cache.mask.shape),
            'fast_color_thres': self.fast_color_thres,
            'density_type': self.density_type,
            'k0_type': self.k0_type,
            'density_config': self.density_config,
            'k0_config': self.k0_config,
            **self.rgbnet_kwargs,
        }
    def load_state_dict(self,param_dict):
        uniq_set = set()
        ps = {}
        stack = []
        def callback(parents, k, v, n):
            stack.append(str(k))
            dc = v.__dict__
            if isinstance(v, nn.ParameterList):
                dc = v.params
            for k2, p in dc.items():
                if isinstance(k2, str) and k2.startswith("_"): continue
                if isinstance(p, jt.Var):
                    if id(p) in uniq_set: continue
                    uniq_set.add(id(p))
                    pname = ".".join(stack[1:]+[str(k2)])
                    if p.requires_grad:
                        dc[k2]=jt.array(param_dict[pname])
                    else:
                        dc[k2]=jt.array(param_dict[pname]).stop_grad()
                    # v.update(p)
                    if len(pname) > len(p.name()):
                        p.name(pname)
        def callback_leave(parents, k, v, n):
            stack.pop()
        self.dfs([], None, callback, callback_leave)
    @jt.no_grad()
    def maskout_near_cam_vox(self, cam_o, near_clip):
        # maskout grid points that between cameras and their near planes
        self_grid_xyz = jt.stack(jt.meshgrid(
            jt.linspace(self.xyz_min[0].item(), self.xyz_max[0].item(), self._world_size[0].item()),
            jt.linspace(self.xyz_min[1].item(), self.xyz_max[1].item(), self._world_size[1].item()),
            jt.linspace(self.xyz_min[2].item(), self.xyz_max[2].item(), self._world_size[2].item()),
        ), -1)
        nearest_dist = jt.stack([
            (self_grid_xyz.unsqueeze(-2) - jt.float32(co)).pow(2).sum(-1).sqrt().min(-1)
            # (self_grid_xyz.unsqueeze(-2) - co).pow(2).sum(-1).sqrt().amin(-1)
            for co in np.split(cam_o,10)  # for memory saving
        ]).min(0)
        
    
        mask=(nearest_dist[None,None] <= near_clip)
        self.density.grid[mask]=-100.
        self.density.grid.requires_grad=True

        # self.density.grid.requires_grad=True
        # sub_grid=self.density.grid[nearest_dist[None,None] <= near_clip]
        # sub_grid.requires_grad=False
        # self.density.grid[nearest_dist[None,None] <= near_clip]=sub_grid
        
        # del nearest_dist
    @jt.no_grad()
    def scale_volume_grid(self, num_voxels):
        print('dvgo: scale_volume_grid start')
        ori_world_size = self._world_size
        self._set_grid_resolution(num_voxels)
        print('dvgo: scale_volume_grid scale world_size from', ori_world_size.tolist(), 'to', self._world_size.tolist())

        self.density.scale_volume_grid(self._world_size)
        self.k0.scale_volume_grid(self._world_size)

        if np.prod(self._world_size.tolist()) <= 256**3:
            self_grid_xyz = jt.stack(jt.meshgrid(
                jt.linspace(self.xyz_min[0].item(), self.xyz_max[0].item(), self._world_size[0].item()),
                jt.linspace(self.xyz_min[1].item(), self.xyz_max[1].item(), self._world_size[1].item()),
                jt.linspace(self.xyz_min[2].item(), self.xyz_max[2].item(), self._world_size[2].item()),
            ), -1)
            self_alpha = nn.max_pool3d(self.activate_density(self.density.get_dense_grid()), kernel_size=3, padding=1, stride=1)[0,0]
            self.mask_cache = grid.MaskGrid(
                    path=None, mask=self.mask_cache(self_grid_xyz) & (self_alpha>self.fast_color_thres),
                    xyz_min=self.xyz_min, xyz_max=self.xyz_max)

        print('dvgo: scale_volume_grid finish')

    @jt.no_grad()
    def update_occupancy_cache(self):
        print("update_occupancy_cache ")
        cache_grid_xyz = jt.stack(jt.meshgrid(
            jt.linspace(self.xyz_min[0].item(), self.xyz_max[0].item(), self.mask_cache.mask.shape[0]),
            jt.linspace(self.xyz_min[1].item(), self.xyz_max[1].item(), self.mask_cache.mask.shape[1]),
            jt.linspace(self.xyz_min[2].item(), self.xyz_max[2].item(), self.mask_cache.mask.shape[2]),
        ), -1)
        cache_grid_density = self.density(cache_grid_xyz)[None,None]
        cache_grid_alpha = self.activate_density(cache_grid_density)
        cache_grid_alpha = nn.max_pool3d(cache_grid_alpha, kernel_size=3, padding=1, stride=1)[0,0]
        self.mask_cache.mask &= (cache_grid_alpha > self.fast_color_thres)

    def voxel_count_views(self, rays_o_tr, rays_d_tr, imsz, near, far, stepsize, downrate=1, irregular_shape=False):
        print('dvgo: voxel_count_views start')
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        eps_time = time.time()

        # N_samples = int(np.linalg.norm(np.array(self.density.shape[2:])+1) / stepsize) + 1
        N_samples = int(np.linalg.norm(np.array(self._world_size.numpy())+1) / stepsize) + 1
        rng = jt.arange(N_samples)[None].float().stop_grad()
        count = jt.zeros_like(self.density.get_dense_grid()).stop_grad()
        #TODO:
        # device = rng.device
        for rays_o_, rays_d_ in zip(rays_o_tr.split(imsz), rays_d_tr.split(imsz)):
            ones = grid.DenseGrid(1, self._world_size, self.xyz_min, self.xyz_max)
            optimizer=jt.optim.SGD([ones.grid],0)
            if irregular_shape:
                rays_o_ = rays_o_.split(10000)
                rays_d_ = rays_d_.split(10000)
            else:
                # rays_o_ = rays_o_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                # rays_d_ = rays_d_[::downrate, ::downrate].to(device).flatten(0,-2).split(10000)
                rays_o_ = rays_o_[::downrate, ::downrate].flatten(0,-2).split(10000)
                rays_d_ = rays_d_[::downrate, ::downrate].flatten(0,-2).split(10000)

            for rays_o, rays_d in zip(rays_o_, rays_d_):
                # print(len(rays_o_), "???")
                #TODO: jittor's where is  not same as pytorch's 
                #
                # vec = jt.where(rays_d==0, jt.full_like(rays_d, 1e-6), rays_d)
                idx=jt.where(rays_d==0)
                vec=rays_d.copy()
                vec[idx]=1e-6
                rate_a = (self.xyz_max - rays_o) / vec
                rate_b = (self.xyz_min - rays_o) / vec
                # t_min = jt.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)
                # t_max = jt.maximum(rate_a, rate_b).amin(-1).clamp(min=near, max=far)
                # jt.clamp()
                t_min = jt.minimum(rate_a, rate_b).max(-1).clamp(min_v=near, max_v=far)
                step = stepsize * self._voxel_size * rng
                interpx = (t_min[...,None] + step/rays_d.norm(dim=-1,keepdim=True))
                rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
                #TODO: backward not supported...... fuck
                optimizer.backward(ones(rays_pts.detach()).sum())
                jt.sync_all()
            with jt.no_grad():
                count = count + (ones.grid.opt_grad(optimizer)> 1)
            # exit(0)
            # del ones
            # del optimizer
            # jt.gc()
            # count.sync(True)
            # # jt.clean_graph()
            # jt.gc()
            
        eps_time = time.time() - eps_time
        print('dvgo: voxel_count_views finish (eps time:', eps_time, 'sec)')
        return count

    def density_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self._world_size.max() / 128
        self.density.total_variation_add_grad(w, w, w, dense_mode)

    def k0_total_variation_add_grad(self, weight, dense_mode):
        w = weight * self._world_size.max() / 128
        self.k0.total_variation_add_grad(w, w, w, dense_mode)

    def activate_density(self, density, interval=None):
        interval = interval if interval is not None else self._voxel_size_ratio
        shape = density.shape
        if density.numel()==0:
            return jt.array([])
        # use object instead of Apply(Function)
        return Raw2Alpha.apply(density.flatten(), self.act_shift, interval).reshape(shape)
        

    def hit_coarse_geo(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Check whether the rays hit the solved coarse geometry or not'''
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        shape = rays_o.shape[:-1]
        #TODO:
        #rays_o = rays_o.reshape(-1, 3).contiguous()
        #rays_d = rays_d.reshape(-1, 3).contiguous()
        # 
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        stepdist = stepsize * self._voxel_size
        #TODO: not implemented
        
        ray_pts, mask_outbbox, ray_id,= render_utils.sample_pts_on_rays(
                rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)[:3]

        # ray_pts, mask_outbbox, ray_id,step_id=jt.Var(np.load("ray_pts.npy")),\
        #                                     jt.Var(np.load("mask_outbbox.npy")),\
        #                                     jt.Var(np.load("ray_id.npy")),\
        #                                     jt.Var(np.load("step_id.npy"))
        #TODO: ~ op not supported
        #mask_inbbox = ~mask_outbbox
        mask_inbbox = (mask_outbbox==False)
        hit = jt.zeros([len(rays_o)], dtype=jt.bool)
        hit[ray_id[mask_inbbox][self.mask_cache(ray_pts[mask_inbbox])]] = 1
        return hit.reshape(shape)

    def sample_ray(self, rays_o, rays_d, near, far, stepsize, **render_kwargs):
        '''Sample query points on rays.
        All the output points are sorted from near to far.
        Input:
            rays_o, rayd_d:   both in [N, 3] indicating ray configurations.
            near, far:        the near and far distance of the rays.
            stepsize:         the number of voxels of each sample step.
        Output:
            ray_pts:          [M, 3] storing all the sampled points.
            ray_id:           [M]    the index of the ray of each point.
            step_id:          [M]    the i'th step on a ray of each point.
        '''
        #TODO: far is re-set 
        far = 1e9  # the given far can be too small while rays stop when hitting scene bbox
        #TODO:
        # rays_o = rays_o.copy()
        # rays_d = rays_d.copy()
        stepdist = stepsize * self._voxel_size
        #TODO: render_utils_cuda.sample_pts_on_rays not implemeted:
        # ray_pts, mask_outbbox, ray_id,step_id=jt.Var(np.load("ray_pts.npy")),\
        #                                     jt.Var(np.load("mask_outbbox.npy")),\
        #                                     jt.array(np.load("ray_id.npy")).int64(),\
        #                                     jt.array(np.load("step_id.npy")).int64()
        ray_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max = render_utils.sample_pts_on_rays(
            rays_o, rays_d, self.xyz_min, self.xyz_max, near, far, stepdist)
        #TODO:
        # bad operand type for unary ~: 'jittor_core.Var'
        # ~ op not supported
        #mask_inbbox = ~mask_outbbox
        mask_inbbox = (mask_outbbox==False)
        ray_pts = ray_pts[mask_inbbox]
        ray_id = ray_id[mask_inbbox]
        step_id = step_id[mask_inbbox]
        return ray_pts, ray_id, step_id

    def execute(self, rays_o, rays_d, viewdirs, global_step=None, **render_kwargs):
        '''Volume rendering
        @rays_o:   [N, 3] the starting point of the N shooting rays.
        @rays_d:   [N, 3] the shooting direction of the N rays.
        @viewdirs: [N, 3] viewing direction to compute positional embedding for MLP.
        '''
        assert len(rays_o.shape)==2 and rays_o.shape[-1]==3, 'Only suuport point queries in [N, 3] format'
        
        ret_dict = {}
        N = len(rays_o)

        # sample points on rays: no problem
        ray_pts, ray_id, step_id = self.sample_ray(
                rays_o=rays_o, rays_d=rays_d, **render_kwargs)
        interval = render_kwargs['stepsize'] * self._voxel_size_ratio

        # skip known free space
        if self.mask_cache is not None:
            mask = self.mask_cache(ray_pts)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for alpha w/ post-activation
        # ray_pts=jt.array(np.load("../DirectVoxGO/ray_pts.npy"))
        # xyz_min=jt.array(np.load("../DirectVoxGO/xyz_min.npy")).stop_grad()
        # xyz_max=jt.array(np.load("../DirectVoxGO/xyz_max.npy")).stop_grad()
        # self.density.xyz_max
        density = self.density(ray_pts.detach()) # no problem
        # alpha2weight has some mismatch with torch (err about 1e-7) # TODO
        alpha = self.activate_density(density, interval)
        if self.fast_color_thres > 0:
            mask = (alpha > self.fast_color_thres)
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]
            density = density[mask]
            alpha = alpha[mask]

        # compute accumulated transmittance
        #TODO:use object instead of apply(Function)
        weights, alphainv_last = Alphas2Weights.apply(alpha, ray_id, N) # weights also different due to alpha

        #weights_bk = alpha * jt.cumprod(jt.concat([jt.ones((alpha.shape[0], 1)), (1.-alpha + 1e-10).reshape((alpha.shape[0], 1))], -1), -1)[:, :-1]
        #print((weights-weights_bk).abs().max())
        if self.fast_color_thres > 0:
            mask = (weights > self.fast_color_thres)
            weights = weights[mask]
            alpha = alpha[mask]
            ray_pts = ray_pts[mask]
            ray_id = ray_id[mask]
            step_id = step_id[mask]

        # query for color
        if self.rgbnet_full_implicit:
            pass
        else:
            #debug
            #TODO:
            # k0=jt.Var(np.load("k0.npy"))
            k0 = self.k0(ray_pts.detach())
        # jt.sync_all()
        if ray_pts.numel()>0:
            if self.rgbnet is None:
                # no view-depend effect
                rgb = jt.sigmoid(k0)
            else:
                # view-dependent color emission
                if self.rgbnet_direct:
                    k0_view = k0
                else:
                    k0_view = k0[:, 3:]
                    k0_diffuse = k0[:, :3]
                viewdirs_emb = (viewdirs.unsqueeze(-1) * self.viewfreq).flatten(-2)
                viewdirs_emb = jt.contrib.concat([viewdirs, viewdirs_emb.sin(), viewdirs_emb.cos()], -1)
                viewdirs_emb = viewdirs_emb.flatten(0,-2)[ray_id]
                rgb_feat = jt.contrib.concat([k0_view, viewdirs_emb], -1)
                rgb_logit = self.rgbnet(rgb_feat)
                if self.rgbnet_direct:
                    rgb = jt.sigmoid(rgb_logit)
                else:
                    rgb = jt.sigmoid(rgb_logit + k0_diffuse)

            # Ray marching
            #TODO: scatter --> segment_coo
            #TODO: whether it need gradient
            # weights=jt.Var(np.load("weights.npy"))
            # ray_id=jt.Var(np.load("ray_id.npy"))
            rgb_marched_t = scatter(x=jt.zeros([N, 3]),dim=0,
                    src=(weights.unsqueeze(-1) * rgb),
                    index=ray_id,
                    reduce='add')
            if render_kwargs.get('rand_bkgd', False) and global_step is not None:
                rgb_marched  = rgb_marched_t+(alphainv_last.unsqueeze(-1) * render_kwargs['rand_bkgd_color'])
            else:
                rgb_marched = rgb_marched_t+(alphainv_last.unsqueeze(-1) * render_kwargs['bg'])
        else:
            rgb=jt.array([]).reshape((0,3))
            rgb_marched_t=jt.array([])
            if render_kwargs.get('rand_bkgd', False) and global_step is not None:
                rgb_marched  = jt.zeros([N, 3])+(alphainv_last.unsqueeze(-1) * render_kwargs['rand_bkgd_color'])
            else:
                rgb_marched=jt.zeros([N, 3])+alphainv_last.unsqueeze(-1) * render_kwargs['bg']
        ret_dict={
            'alphainv_last': alphainv_last,
            'weights': weights,
            'rgb_marched': rgb_marched,
            'raw_alpha': alpha,
            'raw_rgb': rgb,
            'ray_id': ray_id,
        }
        
        if alpha.shape[0]>0:
            if render_kwargs.get('render_depth', False):
                with jt.no_grad():
                    #TODO: 
                    # scatter --> segment_coo
                    depth = scatter( x=jt.zeros([N]),dim=0,
                            src=(weights * step_id),
                            index=ray_id,
                            reduce='add')
                ret_dict.update({'depth': depth})
        else:
            ret_dict.update({'depth': jt.zeros([N])})
        return ret_dict

#TODO:
# if jt.flags.no_grad:
''' Misc
'''
class Raw2Alpha(Function):

    def execute(self, density, shift, interval):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        exp, alpha = render_utils.raw2alpha(density, shift, interval)
        # python_alpha = raw2alpha(density,shift,interval)
        if density.requires_grad:
            self.exp=exp
            self.interval = interval
        return alpha.detach()
    def grad(self, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''
        exp = self.exp
        interval = self.interval
        #return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None
        grad=render_utils.raw2alpha_backward(exp, grad_back, interval)
        #np.save("density_grad0.npy",grad.data)
        return grad, None, None

class Raw2Alpha_nonuni(Function):
    def execute(self, density, shift, interval):
        exp, alpha = render_utils.raw2alpha_nonuni(density, shift, interval)
        if density.requires_grad:
            self.exp=exp
            self.interval = interval
        return alpha
    def grad(self, grad_back):
        exp = self.exp
        interval = self.interval
        # return render_utils_cuda.raw2alpha_nonuni_backward(exp, grad_back.contiguous(), interval), None, None
        return render_utils.raw2alpha_nonuni_backward(exp, grad_back, interval), None, None

class Alphas2Weights(Function):
    def execute(self, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils.alpha2weight(alpha, ray_id, N)
        if alpha.requires_grad:
            #print(alpha.shape)
            self.alpha, self.weights, self.T, self.alphainv_last, self.i_start, self.i_end\
            =alpha.copy(), weights.copy(), T, alphainv_last.copy(), i_start, i_end
            self.n_rays = N
        return weights.detach(), alphainv_last.detach()
    def grad(self, grad_weights, grad_last):
        #jt.sync_all(True)
        alpha, weights, T, alphainv_last, i_start, i_end =\
            self.alpha, self.weights, self.T, self.alphainv_last, self.i_start, self.i_end
        #print(alpha.shape)
        grad_weights.fetch_sync()
        grad_last.fetch_sync()
        grad = render_utils.alpha2weight_backward(
                alpha, weights, T, alphainv_last,
                i_start, i_end, self.n_rays, grad_weights, grad_last)
         
        
        # np.save("grad_weights.npy",grad_weights.data)
        # np.save("grad_last.npy",grad_last.data)
        #np.save("grad.npy",grad.data)
        #grad.detach()
        grad.fetch_sync()
        return grad, None, None


''' Ray and batch
'''
def get_rays(H, W, K, c2w, inverse_y, flip_x, flip_y, mode='center'):

    c2w=jt.float32(c2w)
    # i, j = jt.meshgrid(
    #     jt.linspace(0, W-1, W, device=c2w.device),
    #     jt.linspace(0, H-1, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
    #TODO:
    # bugs: H,W with np.int64 not working  change to int
    H=int(H)
    W=int(W)
    i, j = jt.meshgrid(
        jt.misc.linspace(0, W-1, W),
        jt.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == 'lefttop':
        pass
    elif mode == 'center':
        i, j = i+0.5, j+0.5
    elif mode == 'random':
        i = i+jt.rand_like(i)
        j = j+jt.rand_like(j)
    else:
        raise NotImplementedError

    if flip_x:
        i = i.flip((1,))
    if flip_y:
        j = j.flip((0,))
    if inverse_y:
        dirs = jt.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], jt.ones_like(i)], -1)
    else:
        #dirs = jt.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -jt.ones_like(i)], -1)
        #TODO: jittor cpu stack_op causes unstable result ,produce 0 or nan
        dirs = jt.stack([-(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], jt.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,3], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = jt.stack([o0,o1,o2], -1)
    rays_d = jt.stack([d0,d1,d2], -1)

    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, ndc, inverse_y, flip_x, flip_y, mode='center'):
    rays_o, rays_d = get_rays(H, W, K, c2w, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y, mode=mode)
    
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)
    # jt.norm()
    if ndc:
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)
    return rays_o, rays_d, viewdirs


@jt.no_grad()
def get_training_rays(rgb_tr, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays: start')
    rgb_tr_length = rgb_tr.shape[0]
    assert len(np.unique(HW, axis=0)) == 1
    assert len(np.unique(Ks.reshape(len(Ks),-1), axis=0)) == 1
    assert rgb_tr_length == len(train_poses) and rgb_tr_length == len(Ks) and rgb_tr_length == len(HW)
    H, W = HW[0]
    H=int(H)
    W=int(W)
    K = Ks[0]
    eps_time = time.time()
    #TODO: int64 not supported
    rays_o_tr = jt.zeros([rgb_tr_length, H, W, 3])
    rays_d_tr = jt.zeros([rgb_tr_length,H,  W, 3])
    viewdirs_tr = jt.zeros([rgb_tr_length, H,  W, 3])
    # rays_o_tr = jt.zeros([rgb_tr_length, H, W, 3], device=rgb_tr.device)
    # rays_d_tr = jt.zeros([rgb_tr_length, H, W, 3], device=rgb_tr.device)
    # viewdirs_tr = jt.zeros([rgb_tr_length, H, W, 3], device=rgb_tr.device)
    imsz = [1] * rgb_tr_length
    for i, c2w in enumerate(train_poses):
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc, inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        #TODO: no copy_ and device implemented, use update as an alternative, 
        # rays_o_tr[i].copy_(rays_o.to(rgb_tr.device))
        # rays_d_tr[i].copy_(rays_d.to(rgb_tr.device))
        # viewdirs_tr[i].copy_(viewdirs.to(rgb_tr.device))
        # rays_o_tr[i].update(rays_o)
        # rays_d_tr[i].update(rays_d)
        # viewdirs_tr[i].update(viewdirs)
        rays_o_tr[i]=rays_o
        rays_d_tr[i]=rays_d
        viewdirs_tr[i]=viewdirs
    # rays_o_tr=rays_o_tr.detach()
    # rays_d_tr=rays_d_tr.detach()
    # viewdirs_tr=viewdirs_tr.detach()
        # del rays_o, rays_d, viewdirs
    jt.sync_all(True)
    jt.clean_graph()
    jt.gc()
    eps_time = time.time() - eps_time
    print('get_training_rays: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@jt.no_grad()
def get_training_rays_flatten(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y):
    print('get_training_rays_flatten: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    eps_time = time.time()
    # DEVICE = rgb_tr_ori[0].device
    N = sum(im.shape[0] * im.shape[1] for im in rgb_tr_ori)
    #TODO:
    # rgb_tr = jt.zeros([N,3], device=DEVICE)
    rgb_tr = jt.zeros([N, 3])
    rays_o_tr = jt.zeros_like(rgb_tr)
    rays_d_tr = jt.zeros_like(rgb_tr)
    viewdirs_tr = jt.zeros_like(rgb_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[:2] == (H, W)
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        n = H * W
        rgb_tr[top:top+n] = img.flatten(0,1)
        # rays_o_tr[top:(top + n)].copy_(rays_o.flatten(start_dim=0, end_dim=1).to(DEVICE))
        # rays_d_tr[top:(top + n)].copy_(rays_d.flatten(start_dim=0, end_dim=1).to(DEVICE))
        # viewdirs_tr[top:(top + n)].copy_(viewdirs.flatten(start_dim=0, end_dim=1).to(DEVICE))
        #TODO:
        rays_o_tr[top:(top + n)] = rays_o.flatten(0, 1)
        rays_d_tr[top:(top + n)] = rays_d.flatten(0, 1)
        viewdirs_tr[top:(top + n)] = viewdirs.flatten(0,1)
        imsz.append(n)
        top += n
        
        jt.clean_graph()
        jt.sync_all()
        jt.gc()

    assert top == N
    eps_time = time.time() - eps_time
    print('get_training_rays_flatten: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


@jt.no_grad()
def get_training_rays_in_maskcache_sampling(rgb_tr_ori, train_poses, HW, Ks, ndc, inverse_y, flip_x, flip_y, model, render_kwargs):
    print('get_training_rays_in_maskcache_sampling: start')
    assert len(rgb_tr_ori) == len(train_poses) and len(rgb_tr_ori) == len(Ks) and len(rgb_tr_ori) == len(HW)
    CHUNK = 64
    # DEVICE = rgb_tr_ori[0].device
    # DEVICE = rgb_tr_ori[0].device
    eps_time = time.time()
    N = sum(((im.shape[0] * im.shape[1]) for im in rgb_tr_ori))
    # rgb_tr = jt.zeros([N, 3], device=DEVICE)
    rgb_tr = jt.zeros([N, 4])
    rays_o_tr = jt.zeros([N, 3])
    rays_d_tr = jt.zeros_like(rays_o_tr)
    viewdirs_tr = jt.zeros_like(rays_o_tr)
    imsz = []
    top = 0
    for c2w, img, (H, W), K in zip(train_poses, rgb_tr_ori, HW, Ks):
        assert img.shape[0] == H and img.shape[1] == W
        rays_o, rays_d, viewdirs = get_rays_of_a_view(
                H=H, W=W, K=K, c2w=c2w, ndc=ndc,
                inverse_y=inverse_y, flip_x=flip_x, flip_y=flip_y)
        # mask = jt.empty(img.shape[:2], device=DEVICE, dtype=jt.bool)
        mask = jt.empty(img.shape[:2], dtype=jt.bool)
        for i in range(0, img.shape[0], CHUNK):
            # mask[i:i+CHUNK] = model.hit_coarse_geo(
                    # rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs).to(DEVICE)
            mask[i:i+CHUNK] = model.hit_coarse_geo(
                    rays_o=rays_o[i:i+CHUNK], rays_d=rays_d[i:i+CHUNK], **render_kwargs)
    
        n = mask.sum().item()
        rgb_tr[top:top+n]=img[mask]
        # rays_o_tr[top:(top + n)].copy_(rays_o[mask].to(DEVICE))
        # rays_d_tr[top:(top + n)].copy_(rays_d[mask].to(DEVICE))
        # viewdirs_tr[top:(top + n)].copy_(viewdirs[mask].to(DEVICE))
        rays_o_tr[top:(top + n)]=rays_o[mask]
        rays_d_tr[top:(top + n)]=rays_d[mask]
        viewdirs_tr[top:(top + n)]=viewdirs[mask]
        imsz.append(n)
        top += n
        
        jt.clean_graph()
        jt.sync_all()
        jt.gc()

    print('get_training_rays_in_maskcache_sampling: ratio', top / N)
    rgb_tr = rgb_tr[:top]
    rays_o_tr = rays_o_tr[:top]
    rays_d_tr = rays_d_tr[:top]
    viewdirs_tr = viewdirs_tr[:top]
    eps_time = time.time() - eps_time
    print('get_training_rays_in_maskcache_sampling: finish (eps time:', eps_time, 'sec)')
    return rgb_tr, rays_o_tr, rays_d_tr, viewdirs_tr, imsz


def batch_indices_generator(N, BS):
    # jt.randperm on cuda produce incorrect results in my machine
    idx, top = jt.array(np.random.permutation(N),dtype='int64'), 0
    while True:
        if top + BS > N:
            idx, top = jt.array(np.random.permutation(N),dtype='int64'), 0
        yield idx[top:top+BS]
        top += BS

