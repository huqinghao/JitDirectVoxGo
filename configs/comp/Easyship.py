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
    far=6.,
)  