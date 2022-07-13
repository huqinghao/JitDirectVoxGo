
import jittor as jt
import math

def grid_sampler_3d_forward_cuda(
    input, 
    grid, 
    interpolation_mode, 
    exp_avgpadding_mode_sq, 
    align_corners, 
    ):
    align_corners_txt = "true" if align_corners else "false"
    output = jt.zeros((input.shape[0],input.shape[1],grid.shape[1],grid.shape[2],grid.shape[3]),dtype=input.dtype)
   
    jt.code(inputs=[input,grid],outputs=[output],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;
using jittor::float32;


#define CUDA_KERNEL_LOOP_TYPE(i, n, index_type)  int64_t _i_n_d_e_x = blockIdx.x * blockDim.x + threadIdx.x;  for (index_type i=_i_n_d_e_x; _i_n_d_e_x < (n); _i_n_d_e_x+=blockDim.x * gridDim.x, i=_i_n_d_e_x)

#define CUDA_KERNEL_LOOP(i, n) CUDA_KERNEL_LOOP_TYPE(i, n, int)


namespace{

static __forceinline__ __device__
bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
  return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
}


template<typename scalar_t>
static __forceinline__ __device__
scalar_t safe_downgrade_to_int_range(scalar_t x){
  // -100.0 does not have special meaning. This is just to make sure
  // it's not within_bounds_2d or within_bounds_3d, and does not cause
  // undefined behavior. See #35506.
  if (x > INT_MAX-1 || x < INT_MIN || !::isfinite(static_cast<double>(x)))
    return static_cast<scalar_t>(-100.0);
  return x;
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static __forceinline__ __device__
scalar_t clip_coordinates(scalar_t in, int clip_limit) {
  return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
}


// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static __forceinline__ __device__
scalar_t reflect_coordinates(scalar_t in, int twice_low, int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<scalar_t>(0);
  }
  scalar_t min = static_cast<scalar_t>(twice_low) / 2;
  scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
  in = ::fabs(in - min);
  // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
  scalar_t extra = ::fmod(in, span);
  int flips = static_cast<int>(::floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}


template<typename scalar_t>
static __forceinline__ __device__
scalar_t compute_coordinates(scalar_t coord, int size,
                             int padding_mode,
                             bool align_corners) {
  if (padding_mode == 1) {
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == 2) {
    // reflect coordinates by image borders
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2*(size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2*size - 1);
    }
    // clip coordinates to image borders
    coord = clip_coordinates(coord, size);
  }

  coord = safe_downgrade_to_int_range(coord);
  return coord;
}


template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_unnormalize(scalar_t coord, int size, bool align_corners) {
  if (align_corners) {
    // unnormalize coord from [-1, 1] to [0, size - 1]
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
    return ((coord + 1.f) * size - 1) / 2;
  }
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t>
static __forceinline__ __device__
scalar_t grid_sampler_compute_source_index(
    scalar_t coord,
    int size,
    int padding_mode,
    bool align_corners) {
  coord = grid_sampler_unnormalize(coord, size, align_corners);
  coord = compute_coordinates(coord, size, padding_mode, align_corners);
  return coord;
}


template <typename scalar_t, typename index_t>
__global__ void grid_sampler_3d_kernel(
    const index_t nthreads,
    scalar_t* input,
    scalar_t* grid,
    scalar_t* output,
    int*input_sizes,
    int*grid_sizes,
    int*input_strides,
    int*grid_strides,
    int*output_strides,
    const int interpolation_mode,
    const int padding_mode,
    bool align_corners) {

  index_t C = input_sizes[1];
  index_t inp_D = input_sizes[2];
  index_t inp_H = input_sizes[3];
  index_t inp_W = input_sizes[4];

  index_t out_D = grid_sizes[1];
  index_t out_H = grid_sizes[2];
  index_t out_W = grid_sizes[3];

  index_t inp_sN = input_strides[0];
  index_t inp_sC = input_strides[1];
  index_t inp_sD = input_strides[2];
  index_t inp_sH = input_strides[3];
  index_t inp_sW = input_strides[4];

  index_t grid_sN = grid_strides[0];
  index_t grid_sD = grid_strides[1];
  index_t grid_sH = grid_strides[2];
  index_t grid_sW = grid_strides[3];
  index_t grid_sCoor = grid_strides[4];
  
  index_t out_sN = output_strides[0];
  index_t out_sC = output_strides[1];
  index_t out_sD = output_strides[2];
  index_t out_sH = output_strides[3];
  index_t out_sW = output_strides[4];

  CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
    const index_t w = index % out_W;
    const index_t h = (index / out_W) % out_H;
    const index_t d = (index / (out_H * out_W)) % out_D;
    const index_t n = index / (out_D * out_H * out_W);
    const index_t grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

    // get the corresponding input x, y, z co-ordinates from grid
    scalar_t ix = grid[grid_offset];
    scalar_t iy = grid[grid_offset + grid_sCoor];
    scalar_t iz = grid[grid_offset + 2 * grid_sCoor];

    ix = grid_sampler_compute_source_index(ix, inp_W, padding_mode, align_corners);
    iy = grid_sampler_compute_source_index(iy, inp_H, padding_mode, align_corners);
    iz = grid_sampler_compute_source_index(iz, inp_D, padding_mode, align_corners);

    if (interpolation_mode == 0) { //bilinear
      // get corner pixel values from (x, y, z)
      // for 4d, we used north-east-south-west
      // for 5d, we add top-bottom
      index_t ix_tnw = static_cast<index_t>(::floor(ix));
      index_t iy_tnw = static_cast<index_t>(::floor(iy));
      index_t iz_tnw = static_cast<index_t>(::floor(iz));

      index_t ix_tne = ix_tnw + 1;
      index_t iy_tne = iy_tnw;
      index_t iz_tne = iz_tnw;

      index_t ix_tsw = ix_tnw;
      index_t iy_tsw = iy_tnw + 1;
      index_t iz_tsw = iz_tnw;

      index_t ix_tse = ix_tnw + 1;
      index_t iy_tse = iy_tnw + 1;
      index_t iz_tse = iz_tnw;

      index_t ix_bnw = ix_tnw;
      index_t iy_bnw = iy_tnw;
      index_t iz_bnw = iz_tnw + 1;

      index_t ix_bne = ix_tnw + 1;
      index_t iy_bne = iy_tnw;
      index_t iz_bne = iz_tnw + 1;

      index_t ix_bsw = ix_tnw;
      index_t iy_bsw = iy_tnw + 1;
      index_t iz_bsw = iz_tnw + 1;

      index_t ix_bse = ix_tnw + 1;
      index_t iy_bse = iy_tnw + 1;
      index_t iz_bse = iz_tnw + 1;

      // get surfaces to each neighbor:
      scalar_t tnw = (ix_bse - ix)    * (iy_bse - iy)    * (iz_bse - iz);
      scalar_t tne = (ix    - ix_bsw) * (iy_bsw - iy)    * (iz_bsw - iz);
      scalar_t tsw = (ix_bne - ix)    * (iy    - iy_bne) * (iz_bne - iz);
      scalar_t tse = (ix    - ix_bnw) * (iy    - iy_bnw) * (iz_bnw - iz);
      scalar_t bnw = (ix_tse - ix)    * (iy_tse - iy)    * (iz - iz_tse);
      scalar_t bne = (ix    - ix_tsw) * (iy_tsw - iy)    * (iz - iz_tsw);
      scalar_t bsw = (ix_tne - ix)    * (iy    - iy_tne) * (iz - iz_tne);
      scalar_t bse = (ix    - ix_tnw) * (iy    - iy_tnw) * (iz - iz_tnw);

      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW = output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        //   (c, iz_tnw, iy_tnw, ix_tnw) * tnw + (c, iz_tne, iy_tne, ix_tne) * tne
        // + (c, iz_tsw, iy_tsw, ix_tsw) * tsw + (c, iz_tse, iy_tse, ix_tse) * tse
        // + (c, iz_bnw, iy_bnw, ix_bnw) * bnw + (c, iz_bne, iy_bne, ix_bne) * bne
        // + (c, iz_bsw, iy_bsw, ix_bsw) * bsw + (c, iz_bse, iy_bse, ix_bse) * bse
        *out_ptr_NCDHW = static_cast<scalar_t>(0);
        if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW] * tnw;
        }
        if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW] * tne;
        }
        if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW] * tsw;
        }
        if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW] * tse;
        }
        if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW] * bnw;
        }
        if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW] * bne;
        }
        if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW] * bsw;
        }
        if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW += inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW] * bse;
        }
      }
    } else if (interpolation_mode == 1 ) { // nearest
      index_t ix_nearest = static_cast<index_t>(::round(ix));
      index_t iy_nearest = static_cast<index_t>(::round(iy));
      index_t iz_nearest = static_cast<index_t>(::round(iz));

      // assign nearest neighor pixel value to output pixel
      auto inp_ptr_NC = input + n * inp_sN;
      auto out_ptr_NCDHW = output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;
      for (index_t c = 0; c < C; ++c, inp_ptr_NC += inp_sC, out_ptr_NCDHW += out_sC) {
        if (within_bounds_3d(iz_nearest, iy_nearest, ix_nearest, inp_D, inp_H, inp_W)) {
          *out_ptr_NCDHW = inp_ptr_NC[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCDHW = static_cast<scalar_t>(0);
        }
      }
    }
  }
}


}
   
    
    
    ''',
    cuda_src=f'''
    @alias(input, in0)
    @alias(grid, in1)
    @alias(output, out0)
    
    thrust :: host_vector <int > host_input_sizes (5);
    host_input_sizes[0]=input_shape0;
    host_input_sizes[1]=input_shape1;
    host_input_sizes[2]=input_shape2;
    host_input_sizes[3]=input_shape3;
    host_input_sizes[4]=input_shape4;
    
    thrust :: device_vector <int > input_sizes=host_input_sizes;
    int*input_sizes_ptr=thrust::raw_pointer_cast(&input_sizes[0]);
    
    thrust :: host_vector <int > host_grid_sizes (5);
    host_grid_sizes[0]=grid_shape0;
    host_grid_sizes[1]=grid_shape1;
    host_grid_sizes[2]=grid_shape2;
    host_grid_sizes[3]=grid_shape3;
    host_grid_sizes[4]=grid_shape4;
    
    
    thrust :: device_vector <int > grid_sizes=host_grid_sizes;
    int*grid_sizes_ptr=thrust::raw_pointer_cast(&grid_sizes[0]);

    index_t inp_sW = 1;
    index_t inp_sH = input_shape4;
    index_t inp_sD = inp_sH*input_shape3;
    index_t inp_sC = inp_sD*input_shape2;
    index_t inp_sN = inp_sC*input_shape1;
    thrust :: host_vector <int > host_input_strides (5);
    host_input_strides[0]=inp_sN;
    host_input_strides[1]=inp_sC;
    host_input_strides[2]=inp_sD;
    host_input_strides[3]=inp_sH;
    host_input_strides[4]=inp_sW;
    thrust :: device_vector <int > input_strides=host_input_strides;
    int*input_strides_ptr=thrust::raw_pointer_cast(&input_strides[0]);

    index_t grid_sCoor =1;
    index_t grid_sW = grid_sCoor*grid_shape4;
    index_t grid_sH = grid_sW*grid_shape3;
    index_t grid_sD = grid_sH*grid_shape2;
    index_t grid_sN = grid_sD* grid_shape1;

    thrust ::host_vector <int > host_grid_strides (5);
    host_grid_strides[0]=grid_sN;
    host_grid_strides[1]=grid_sD;
    host_grid_strides[2]=grid_sH;
    host_grid_strides[3]=grid_sW;
    host_grid_strides[4]=grid_sCoor;
    thrust::device_vector <int > grid_strides=host_grid_strides;
    int*grid_strides_ptr=thrust::raw_pointer_cast(&grid_strides[0]);
  
    index_t out_sW = 1;
    index_t out_sH = out_sW*output_shape4;
    index_t out_sD = out_sH*output_shape3;
    index_t out_sC = out_sD*output_shape2;
    index_t out_sN = out_sC*output_shape1;
    thrust::host_vector <int > host_output_strides (5);

    host_output_strides[0]=out_sN;
    host_output_strides[1]=out_sC;
    host_output_strides[2]=out_sD;
    host_output_strides[3]=out_sH;
    host_output_strides[4]=out_sW;
    thrust::device_vector <int > output_strides=host_output_strides;
    int*output_strides_ptr=thrust::raw_pointer_cast(&output_strides[0]);

    auto N = input_shape0;
    auto D = grid_shape1;
    auto H = grid_shape2;
    auto W = grid_shape3;
    int64_t count = N * D * H * W;
    
    if(count > 0) {{
      const int num_threads = 512;
      const int blocks = (count + num_threads - 1 ) / num_threads;
      
      grid_sampler_3d_kernel<float32><<<blocks, num_threads>>>(
        count,
        input_p,
        grid_p,
        output_p,
        input_sizes_ptr,
        grid_sizes_ptr,
        input_strides_ptr,
        grid_strides_ptr,
        output_strides_ptr,
        {interpolation_mode},
        {exp_avgpadding_mode_sq},
        true
      );

    }}
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
      printf("Error in grid sampler forward: %s\\n", cudaGetErrorString(err));
        
    ''')
    
    return output



    
    
    