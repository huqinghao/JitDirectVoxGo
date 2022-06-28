from jittor import Function


def interpolate(X, size=None, scale_factor=None, mode='bilinear', align_corners=False, tf_mode=False):
    if scale_factor is not None:
        size = [int(X.shape[-2] * scale_factor), int(X.shape[-1] * scale_factor)]
    if isinstance(size, int):
        size = (size, size)
    if scale_factor is not None and scale_factor > 1:
        return up_sample3d(X, size, mode, align_corners, tf_mode)
    else:
        print("not support")

def up_sample3d(
    input,
    size,
    mode: str = "trilinear",
    align_corners: bool = None,
    tf_mode=False,
) :
    if mode != "trilinear" :
        raise ValueError(
            "nn.functional.grid_sample(): expected mode to be "
            "'bilinear', 'nearest' or 'bicubic', but got: '{}'".format(mode))


    if align_corners is None:
        print(
            "Default grid_sample and affine_grid behavior has changed "
            "to align_corners=False since 1.3.0. Please specify "
            "align_corners=True if the old behavior is desired. "
            "See the documentation of grid_sample for details."
        )
        align_corners = False

    return UpSampler3d()(input, size, align_corners)

from jittor import Function

class UpSampler3d(Function):

    def execute(self, input, size, align_corners):
        '''
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        '''
        # exp, alpha = render_utils.raw2alpha(density, shift, interval)
        
        return sampled_grid

    def grad(self, grad_back):
        '''
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        '''

        #return render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval), None, None

        return  grid_grads,None,None
