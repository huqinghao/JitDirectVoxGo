# change stepsize and N_iters, num_voxels, net width, net_depth

_base_ = '../default.py'
basedir = './logs/nerf_synthetic'

data = dict(
    datadir='./data/nerf_synthetic/Easyship',
    npy_datadir='./raw_npy_data/nerf_synthetic/Easyship/',
    dataset_type='blender',
    white_bkgd=True,
    rand_bkgd=False,
    half_res=False,
    near=0.5,
    far=15,
) 


coarse_model_and_render = dict(
    stepsize=0.5,
)
fine_model_and_render = dict(
    num_voxels=192**3,
    num_voxels_base=192**3,
    stepsize=0.5, 
    rgbnet_dim=12,
    rgbnet_depth=4,               # depth of the colors MLP (there are rgbnet_depth-1 intermediate features)
    rgbnet_width=256,
)


coarse_train = dict(
    N_iters=5000,                 # number of optimization steps
    # N_rand=8192,                  # batch size (number of random rays per optimization step)
    # lrate_density=1e-1,           # lr of density voxel grid
    # lrate_k0=1e-1,                # lr of color/feature voxel grid
    lrate_decay=20,
)
fine_train=dict(
    N_iters=40000,
    # pervoxel_lr=False,
    # ray_sampler='random',
    # weight_entropy_last=0.001,
    # weight_rgbper=0.01,
    lrate_decay=20,
    pg_scale=[1000, 2000, 3000, 4000, 6000],
    # skip_zero_grad_fields=['density', 'k0'],
)



ratio = 2
expname = f'Easyship4_{data["near"]}_{data["far"]}'
fine_train["N_iters"] *= ratio
coarse_train["N_iters"] *= ratio
fine_train["pg_scale"] = [ratio * step for step in fine_train["pg_scale"]]
coarse_train["lrate_decay"] *= ratio
fine_train["lrate_decay"] *= ratio
