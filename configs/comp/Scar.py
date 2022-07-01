_base_ = '../default.py'
expname = 'ScarBaseline36'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/Scar',
    dataset_type='blender',
    white_bkgd=True,
    half_res=False,
    near=0.5,
    far=6,
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

