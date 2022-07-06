from jittor import Function
import jittor as jt
from .grid_sampler_backward import grid_sampler_3d_backward_cuda
from .grid_sampler_cuda import grid_sampler_3d_forward_cuda
def grid_sample(
    input,
    grid,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = None,
) :
    if mode != "bilinear" and mode != "nearest" and mode != "bicubic":
        raise ValueError(
            "nn.functional.grid_sample(): expected mode to be "
            "'bilinear', 'nearest' or 'bicubic', but got: '{}'".format(mode)
        )
    if padding_mode != "zeros" and padding_mode != "border" and padding_mode != "reflection":
        raise ValueError(
            "nn.functional.grid_sample(): expected padding_mode "
            "to be 'zeros', 'border', or 'reflection', "
            "but got: '{}'".format(padding_mode)
        )

    if mode == "bilinear":
        mode_enum = 0
    elif mode == "nearest":
        mode_enum = 1
    else:  # mode == 'bicubic'
        mode_enum = 2

    if padding_mode == "zeros":
        padding_mode_enum = 0
    elif padding_mode == "border":
        padding_mode_enum = 1
    else:  # padding_mode == 'reflection'
        padding_mode_enum = 2

    if align_corners is None:
        print(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False
        
    align_corners = 1 if align_corners else 0
    
    return GridSampler()(input, grid, mode_enum, padding_mode_enum, align_corners)

from jittor import Function

class GridSampler(Function):

    def execute(self, input, grid, mode_enum, padding_mode_enum, align_corners):

        assert len(input.shape) == 5 , "only support grid sampler 3-D, which need 5d input "
        assert len(grid.shape) == 5, "only support grid sampler 3-D, which need 5d grid"
        
        self.input,self.grid,self.mode_enum,self.padding_mode_enum,self.align_corners=\
            input,grid,mode_enum,padding_mode_enum,align_corners

        output = grid_sampler_3d_forward_cuda(input, grid,mode_enum, padding_mode_enum, align_corners)
        return output
        
    def grad(self, grad_output):

        grad_input, grad_grid = grid_sampler_3d_backward_cuda(grad_output,self.input,self.grid,self.mode_enum,self.padding_mode_enum,self.align_corners)
        
        return grad_input, grad_grid, None, None, None
    
if __name__ == '__main__':
    import numpy as np
    import torch
    jt.flags.use_cuda = 2 
    # jt.flags.use_device = 7
    grid = np.random.randn(1,3,99,101,101)
    xyz = np.random.randn(629236,3)
    xyz_min = np.min(xyz,axis=0)
    xyz_max = np.max(xyz,axis=0)
    nor_xyz = np.flip(((xyz - xyz_min) / (xyz_max - xyz_min)),-1) * 2 - 1
    nor_xyz = nor_xyz.reshape(1,1,1,-1,3)
    
    jt_grid = jt.array(grid)
    jt_nor_xyz = jt.array(nor_xyz)
        
    jt_feature = jt.nn.grid_sample(jt_grid, jt_nor_xyz, mode='bilinear', align_corners=True) 
    manu_feature = grid_sample(jt_grid,jt_nor_xyz, mode='bilinear', align_corners=True) 
    jt.sync_all()
    
    cpu_gpu_diff = manu_feature - jt_feature
    
    torch_grid = torch.from_numpy(grid)
    torch_nor_xyz = torch.from_numpy(nor_xyz)
    
    torch_self_diff = torch.nn.functional.grid_sample(torch_grid,torch_nor_xyz,mode='bilinear',align_corners=True).numpy() - manu_feature.data


    import time 
    start_time = time.time()
    total_count = 2000
    for i in range(total_count):
      manu_feature = grid_sample(jt_grid,jt_nor_xyz, mode='bilinear', align_corners=True) 
      jt.sync_all()
    total_time = time.time() - start_time
    print("AVG time : ", total_time/total_count)