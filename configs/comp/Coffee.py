_base_ = '../default.py'
expname = 'CoffeeBaseline7'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/Coffee',
    npy_datadir='./raw_npy_data/nerf_synthetic/coffee/',
    dataset_type='blender',
    white_bkgd=True,
    rand_bkgd=True,
    half_res=False,
    near=0.5,
    far=16,
)  

coarse_train = dict(
    N_iters=5000,                 # number of optimization steps
    # N_rand=8192,                  # batch size (number of random rays per optimization step)
    # lrate_density=1e-1,           # lr of density voxel grid
    # lrate_k0=1e-1,                # lr of color/feature voxel grid
)
fine_train=dict(
    N_iters=20000,
    # pervoxel_lr=False,
    # ray_sampler='random',
    # weight_entropy_last=0.001,
    # weight_rgbper=0.01,
    pg_scale=[1000, 2000, 3000, 4000],
    # skip_zero_grad_fields=['density', 'k0'],
)


ratio = 2
expname = f'Random_Coffee_{data["near"]}_{data["far"]}'
fine_train["N_iters"] *= ratio
coarse_train["N_iters"] *= ratio
fine_train["pg_scale"] *= ratio