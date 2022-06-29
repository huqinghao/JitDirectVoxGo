from jittor import Function
import jittor as jt

def interpolate(X, size=None, scale_factor=None, mode='trilinear', align_corners=False, tf_mode=False):
    
    if mode != "trilinear":
        return jt.nn.interpolate(X,size,scale_factor,mode,align_corners,tf_mode)
    else:
        return up_sample3d(X, size, mode, align_corners, tf_mode)

def up_sample3d(
    input,
    size,
    mode: str = "trilinear",
    align_corners: bool = None,
    tf_mode=False,
):
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
              
        return up_sampler_3d_forward_cuda(input,size,align_corners)


def up_sampler_3d_forward_cuda(
    input, 
    output_size, 
    align_corners
    ):
    if align_corners == False:
        raise ValueError("Not support Align Corner set False Now!")
    
    output = jt.zeros([i for i in input.shape[:2]] + [i.item() for i in output_size])
    return jt.code(inputs=[input],outputs=[output],
    cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

namespace{

template <typename scalar_t, typename accscalar_t>
__global__ void upsample_trilinear3d_out_frame(
    const int n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const std::vector<int> input_shape,
    const std::vector<int> output_shape,
    const scalar_t* __restrict__ idata,
    scalar_t* __restrict__ odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = input_shape[0]; 
  const int channels = input_shape[1]; 
  const int depth1 = input_shape[2]; 
  const int height1 = input_shape[3]; 
  const int width1 = input_shape[4]; 
  const int depth2 = output_shape[2];
  const int height2 = output_shape[3];
  const int width2 = output_shape[4];

  if (index < n) {
    const int w2 = (index % (height2 * width2)) %  width2; 
    const int h2 = (index % (height2 * width2)) /  width2; 
    const int t2 = index / (height2 * width2); 
    
    if (depth1 == depth2 && height1 == height2 && width1 == width2) {
      const int t1 = t2;
      const int h1 = h2;
      const int w1 = w2;

      for (int n = 0; n < batchsize; n++) {
        for (int c = 0; c < channels; ++c) {
          const scalar_t val = idata[n][c][t1][h1][w1];
          odata[n][c][t2][h2][w2] = val;
        }
      }
      return;
    }
    
    const accscalar_t t1r = rdepth * t2;
    const int t1 = t1r;
    const int t1p = (t1 < depth1 - 1) ? 1 : 0;
    const accscalar_t t1lambda = t1r - t1;
    const accscalar_t t0lambda = static_cast<accscalar_t>(1) - t1lambda;
    
    const accscalar_t h1r = rheight * h2;
    const int h1 = h1r;
    const int h1p = (h1 < height1 - 1) ? 1 : 0;
    const accscalar_t h1lambda = h1r - h1;
    const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
    
    const accscalar_t w1r = rwidth * w2;
    const int w1 = w1r;
    const int w1p = (w1 < width1 - 1) ? 1 : 0;
    const accscalar_t w1lambda = w1r - w1;
    const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;
    
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; ++c) {
        const accscalar_t val = t0lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1][h1][w1] +
                      w1lambda * idata[n][c][t1][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1][h1 + h1p][w1 + w1p])) +
            t1lambda *
                (h0lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1][w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n][c][t1 + t1p][h1 + h1p][w1] +
                      w1lambda * idata[n][c][t1 + t1p][h1 + h1p][w1 + w1p]));
        odata[n][c][t2][h2][w2] = static_cast<scalar_t>(val);
      }
    }
  }
}   
}
    
    ''',
    cuda_src='''
    @alias(input, in0)
    @alias(output, out0)
    
    const int output_depth = output_shape2;
    const int output_height = output_shape3;
    const int output_width = output_shape4;
    
    const int input_depth = input_shape2;
    const int input_height = input_shape3;
    const int input_width = input_shape4;
    
    const int batch_size = input_shape0;
    const int channel_size = input_shape1;
    
    const std::vector<int> in_shape = ({batch_size, channel_size, input_depth, input_height, input_width});
    const std::vector<int> out_shape = ({batch_size, channel_size, output_depth, output_height, output_width});
    
    const int num_kernels = output_depth * output_height * output_width;
    const int num_threads = 512;
    const int blocks = (num_kernels + num_threads - 1 ) / num_threads;
    
    using accscalar_t = float32;
    const accscalar_t rdepth = (input_depth - 1) / (output_depth - 1);
    const accscalar_t rheight = (input_height - 1) / (output_height - 1);
    const accscalar_t rwidth = (input_width - 1) / (output_width - 1);
    
    upsample_trilinear3d_out_frame<float32,accscalar_t><<<blocks, num_threads>>>(
        num_kernels,rdepth,rheight,rwidth,in_shape,out_shape,input_p,output_p
    );
    
    
    ''')
