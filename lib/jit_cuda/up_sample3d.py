from jittor import Function
import jittor as jt
import torch
import torch.nn.functional as F

def interpolate(X, size=None, scale_factor=None, mode='trilinear', align_corners=False, tf_mode=False):
    
    if mode != "trilinear":
        return jt.nn.interpolate(X,size,scale_factor,mode,align_corners,tf_mode)
    else:
        return UpSampler3d()(X, size, align_corners)

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
    jt.code(inputs=[input],outputs=[output],
    cuda_header='''
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;
using jittor::float32;

namespace{

template <typename scalar_t, typename accscalar_t>
__global__ void upsample_trilinear3d_out_frame(
    const int n,
    const accscalar_t rdepth,
    const accscalar_t rheight,
    const accscalar_t rwidth,
    const int batch_size, const int channel_size, const int input_depth, const int input_height, const int input_width,
    const int output_depth, const int output_height, const int output_width,
    const scalar_t* __restrict__ idata,
    scalar_t* __restrict__ odata) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  const int batchsize = batch_size; 
  const int channels = channel_size; 
  const int depth1 = input_depth; 
  const int height1 = input_height; 
  const int width1 = input_width; 
  const int depth2 = output_depth;
  const int height2 = output_height;
  const int width2 = output_width;

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
        //  const scalar_t val = idata[n][c][t1][h1][w1];
        // odata[n][c][t2][h2][w2] = val;
        
        const int in_index = n*channels*depth1*height1*width1 + c*depth1*height1*width1 + t1*height1*width1 + h1*width1 + w1 ;
        const int out_index = n*channels*depth2*height2*width2 + c*depth2*height2*width2 + t2*height2*width2 + h2*width2 + w2 ;
        const scalar_t val = idata[in_index];
        odata[out_index] = val;
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

        // const accscalar_t val = t0lambda *
        //         (h0lambda *
        //              (w0lambda * idata[n][c][t1][h1][w1] +
        //               w1lambda * idata[n][c][t1][h1][w1 + w1p]) +
        //          h1lambda *
        //              (w0lambda * idata[n][c][t1][h1 + h1p][w1] +
        //               w1lambda * idata[n][c][t1][h1 + h1p][w1 + w1p])) +
        //     t1lambda *
        //         (h0lambda *
        //              (w0lambda * idata[n][c][t1 + t1p][h1][w1] +
        //               w1lambda * idata[n][c][t1 + t1p][h1][w1 + w1p]) +
        //          h1lambda *
        //              (w0lambda * idata[n][c][t1 + t1p][h1 + h1p][w1] +
        //               w1lambda * idata[n][c][t1 + t1p][h1 + h1p][w1 + w1p]));
        // odata[n][c][t2][h2][w2] = static_cast<scalar_t>(val);
        
        // const int in_index = n*channels*depth1*height1*width1 + c*depth1*height1*width1 + t1*height1*width1 + h1*width1 + w1 ;
        
        const accscalar_t val = t0lambda *
                (h0lambda *
                     (w0lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + t1*height1*width1 + h1*width1 + w1] +
                      w1lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + t1*height1*width1 + h1*width1 + w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + t1*height1*width1 + (h1+h1p)*width1 + w1] +
                      w1lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + t1*height1*width1 + (h1+h1p)*width1 + w1 + w1p])) +
            t1lambda *
                (h0lambda *
                     (w0lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + (t1+t1p)*height1*width1 + h1*width1 + w1] +
                      w1lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + (t1+t1p)*height1*width1 + h1*width1 + w1 + w1p]) +
                 h1lambda *
                     (w0lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + (t1+t1p)*height1*width1 + (h1+h1p)*width1 + w1] +
                      w1lambda * idata[n*channels*depth1*height1*width1 + c*depth1*height1*width1 + (t1+t1p)*height1*width1 + (h1+h1p)*width1 + w1 + w1p]));
        
        
        const int out_index = n*channels*depth2*height2*width2 + c*depth2*height2*width2 + t2*height2*width2 + h2*width2 + w2;
        odata[out_index] = static_cast<scalar_t>(val);


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
    
    const int num_kernels = output_depth * output_height * output_width;
    const int num_threads = 512;
    const int blocks = (num_kernels + num_threads - 1 ) / num_threads;
    
    const float32 rdepth = (input_depth - 1)*1.0 / (output_depth - 1);
    const float32 rheight = (input_height - 1)*1.0 / (output_height - 1);
    const float32 rwidth = (input_width - 1)*1.0 / (output_width - 1);
    
    upsample_trilinear3d_out_frame<float32,float32><<<blocks, num_threads>>>(
        num_kernels,rdepth,rheight,rwidth,
        batch_size, channel_size,input_depth,input_height,input_width,
        output_depth,output_height,output_width,
        input_p,output_p);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in upsample3d: %s\\n", cudaGetErrorString(err));
        
    ''')
    
    return output



if __name__ == '__main__':
    import numpy as np 
    jt.flags.use_cuda = 2
    input = np.random.rand(1,1,89,100,120).astype("float32").clip(0.1,0.8)
    np.save("npy/interpolate_input.npy",input)
    
    input_jittor = jt.array(input)
    res_jittor = interpolate(input_jittor,jt.array((130,248,355)),mode='trilinear',align_corners=True)
    
    res_torch = F.interpolate(torch.from_numpy(input),(130,248,355),mode='trilinear',align_corners=True)
        
    diff = res_torch.numpy() - res_jittor.data
    print(diff)
    
    
    print(diff.shape)
    
    
    