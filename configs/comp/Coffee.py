_base_ = '../default.py'
expname = 'CoffeeBaseline7'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/Coffee',
    npy_datadir='data/nerf_synthetic/coffee/',
    dataset_type='blender',
    white_bkgd=True,
    half_res=False,
    near=0.5,
    far=16.,
)  
coarse_train = dict(
    N_iters=10000,                 # number of optimization steps
    N_rand=8192,                  # batch size (number of random rays per optimization step)
    lrate_density=1e-1,           # lr of density voxel grid
    lrate_k0=1e-1,                # lr of color/feature voxel grid
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

# fine_model_and_render=dict(
#     bbox_thres=1e-6,              # threshold to determine known free-space in the fine stage
#     mask_cache_thres=1e-3,        # threshold to determine a tighten BBox in the fine stage
#     fast_color_thres=1e-3,
# )