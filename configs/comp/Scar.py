_base_ = '../default.py'
expname = 'ScarBaseline36'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/Scar',
    dataset_type='blender',
    white_bkgd=True,
    half_res=False,
    near=0.5,
    far=20.,
)  
coarse_train = dict(
    N_iters=10000,                 # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
    # lrate_decay=4,               # lr decay by 0.1 after every lrate_decay*1000 steps
    # weight_main=1.0,              # weight of photometric loss
    # weight_entropy_last=0.01,     # weight of background entropy loss
    # weight_rgbper=0.1,            # weight of per-point rgb loss
)
fine_train=dict(
    N_iters=40000,
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000, 4000],
    skip_zero_grad_fields=['density', 'k0'],
)
# coarse_model_and_render = dict(
#     # mask_cache_thres=1e-6,        # threshold to determine a tighten BBox in the fine stage
#     # rgbnet_dim=0,                 # feature voxel grid dim
#     # rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
#     # rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
#     # rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
#     rgbnet_width=128,             # width of the colors MLP
#     # alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
#     # fast_color_thres=1e-7,        # threshold of alpha value to skip the fine stage sampled point
#     # maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
#     bbox_thres=1e-4,              # threshold to determine known free-space in the fine stage
#     mask_cache_thres=1e-4,        # threshold to determine a tighten BBox in the fine stage
#     stepsize=0.5,                 # sampling stepsize in volume rendering
#     num_voxels=1024000,
#     # rgbnet_full_implicit=True,   # let the colors MLP ignore feature voxel grid

#     # fast_color_thres=1e-6,
#     num_voxels_base=1024000*5,      # to rescale delta distance
#     # world_bound_scale=1.
# )

# fine_model_and_render=dict(
#     bbox_thres=1e-6,              # threshold to determine known free-space in the fine stage
#     # mask_cache_thres=1e-5,        # threshold to determine a tighten BBox in the fine stage
#     fast_color_thres=1e-4,
#     rgbnet_width=128,             # width of the colors MLP
#     num_voxels=160**3,
#     num_voxels_base=160**3,      # to rescale delta distance
# )
