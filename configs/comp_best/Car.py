_base_ = '../default.py'
expname = 'CarWider'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/Car',
    dataset_type='blender',
    white_bkgd=True,
    half_res=False,
    near=6,
    far=20,
)  
coarse_train = dict(
    N_iters=10000,                 # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
)
fine_train=dict(
    N_iters=40000,
    N_rand=8192//2,                  # batch size (number of random rays per optimization step)
    pervoxel_lr=False,
    ray_sampler='in_maskcache',
    weight_entropy_last=0.001,
    weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000, 4000],
    skip_zero_grad_fields=['density', 'k0'],
)
coarse_model_and_render = dict(
    num_voxels=1024000,           # expected number of voxel
    num_voxels_base=1024000,      # to rescale delta distance
    mpi_depth=128,                # the number of planes in Multiplane Image (work when ndc=True)
    nearest=False,                # nearest interpolation
    pre_act_density=False,        # pre-activated trilinear interpolation
    in_act_density=False,         # in-activated trilinear interpolation
    bbox_thres=1e-3,              # threshold to determine known free-space in the fine stage
    mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
    rgbnet_dim=0,                 # feature voxel grid dim
    rgbnet_full_implicit=False,   # let the colors MLP ignore feature voxel grid
    rgbnet_direct=True,           # set to False to treat the first 3 dim of feature voxel grid as diffuse rgb
    rgbnet_depth=3,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=128,             # width of the colors MLP
    alpha_init=1e-6,              # set the alpha values everywhere at the begin of training
    fast_color_thres=1e-7,        # threshold of alpha value to skip the fine stage sampled point
    maskout_near_cam_vox=True,    # maskout grid points that between cameras and their near planes
    world_bound_scale=1,          # rescale the BBox enclosing the scene
)

fine_model_and_render =dict(
    num_voxels=160**3,
    num_voxels_base=160**3,
    rgbnet_dim=128,
    alpha_init=1e-2,
    rgbnet_depth=6,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=384,           
    fast_color_thres=1e-4,
    maskout_near_cam_vox=False,
    world_bound_scale=1.05,
)