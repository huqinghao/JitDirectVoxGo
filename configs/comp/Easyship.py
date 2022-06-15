_base_ = '../default.py'
expname = 'Easyship'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/Easyship',
    dataset_type='blender',
    white_bkgd=True,
    rand_bkgd=False,
    half_res=False,
    near=0.5,
    far=20.,
)  
# coarse_train = dict(
#     N_iters=5000,                 # number of optimization steps
#     N_rand=8192,                  # batch size (number of random rays per optimization step)
#     lrate_density=1e-1,           # lr of density voxel grid
#     lrate_k0=1e-1,                # lr of color/feature voxel grid
# )
# fine_train=dict(
#     N_iters=20000,
#     pervoxel_lr=False,
#     ray_sampler='in_maskcache',
#     weight_entropy_last=0.001,
#     weight_rgbper=0.01,
#     pg_scale=[1000, 2000, 3000, 4000],
#     skip_zero_grad_fields=['density', 'k0'],
# )
