
import jittor as jt
import math  

# const TensorBase &output, const TensorBase &input, const TensorBase &grid,
#     int64_t interpolation_mode, int64_t padding_mode, bool align_corners

# Tensor grid_sampler_3d_cuda(const Tensor& input, const Tensor& grid,
#                             int64_t interpolation_mode, int64_t padding_mode,
#                             bool align_corners) {
#   auto in_size = input.sizes();
#   auto grid_size = grid.sizes();
#   auto output = at::empty(
#       {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
#       input.options());
#   launch_grid_sampler_3d_forward_kernel(
#       output, input, grid, interpolation_mode, padding_mode, align_corners);
#   return output;
# }

def grid_sampler_3d_forward_cuda(
    input, 
    grid, 
    interpolation_mode, 
    exp_avgpadding_mode_sq, 
    align_corners, 
    ):
    #   auto in_size = input.sizes();
#   auto grid_size = grid.sizes();
#   auto output = at::empty(
#       {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
#       input.options());
   
    return jt.code(inputs=[input,grid],outputs=[param,exp_avg, exp_avg_sq],
    cuda_header=''',
    cuda_src=''')


# std::tuple<Tensor, Tensor>
# grid_sampler_3d_backward_cuda(const Tensor& grad_output, const Tensor& input,
#                               const Tensor& grid, int64_t interpolation_mode, int64_t padding_mode,
#                               bool align_corners, std::array<bool,2> output_mask) {
#   auto input_requires_grad = output_mask[0];
#   Tensor grad_input = ([&]() {
#     if (input_requires_grad) {
#       return at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
#     } else {
#       return Tensor();
#     }
#   })();
#   auto grad_grid = at::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
#   launch_grid_sampler_3d_backward_kernel(
#       grad_input, grad_grid, grad_output, input,
#       grid, interpolation_mode, padding_mode, align_corners, output_mask);
#   return std::make_tuple(grad_input, grad_grid);
# }

def grid_sampler_3d_backward_cuda(
    input, 
    grid, 
    interpolation_mode, 
    exp_avgpadding_mode_sq, 
    align_corners, 
    ):
    #   auto in_size = input.sizes();
#   auto grid_size = grid.sizes();
#   auto output = at::empty(
#       {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]},
#       input.options());
   
    return jt.code(inputs=[input,grid],outputs=[param,exp_avg, exp_avg_sq],
    cuda_header=''',
    cuda_src=''')