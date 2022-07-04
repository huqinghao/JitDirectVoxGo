from jittor import Function
from grid_sampler_backward import grid_sampler_3d_backward_cuda
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

    return GridSampler()(input, grid, mode_enum, padding_mode_enum, align_corners)

from jittor import Function

class GridSampler(Function):

    def execute(self, input, grid, mode_enum, padding_mode_enum, align_corners):

        self.input,self.grid,self.mode_enum,self.padding_mode_enum,self.align_corners=\
            input,grid,mode_enum,padding_mode_enum,align_corners
        # exp, alpha = render_utils.raw2alpha(density, shift, interval)
        
        return None

    def grad(self, grad_output):
        #return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None
        grad_input, grad_grid=grid_sampler_3d_backward_cuda(grad_output, self.input,self.grid,self.mode_enum,self.padding_mode_enum,self.align_corners)
        return grad_input, grad_grid,None,None,None,