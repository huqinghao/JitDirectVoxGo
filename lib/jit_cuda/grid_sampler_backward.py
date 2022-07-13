
import jittor as jt
import math  


def grid_sampler_3d_backward_cuda(
    grad_output, input, grid, interpolation_mode, padding_mode, align_corners,input_requires_grad=True
    ):
    # interpolation_mode
    if input_requires_grad:
        grad_input=jt.zeros_like(input)
    else:
        grad_input =None,

    grad_grid = jt.zeros(grid.shape)

    # t::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);

#   launch_grid_sampler_3d_backward_kernel(
#       grad_input, grad_grid, grad_output, input,
#       grid, interpolation_mode, padding_mode, align_corners, output_mask);
#   return std::make_tuple(grad_input, grad_grid);
   
    
#
    jt.code(inputs=[grad_output, input, grid],outputs=[grad_input,grad_grid],
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
    static __forceinline__ __device__
  bool within_bounds_3d(int d, int h, int w, int D, int H, int W) {
    return d >= 0 && d < D && h >= 0 && h < H && w >= 0 && w < W;
  }
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t reflect_coordinates_set_grad(scalar_t in, int twice_low, int twice_high,
                                        scalar_t *grad_in) {
    if (twice_low == twice_high) {
      *grad_in = static_cast<scalar_t>(0);
      return static_cast<scalar_t>(0);
    }
    int grad_in_mult_;
    scalar_t min = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span = static_cast<scalar_t>(twice_high - twice_low) / 2;
    in = in - min;
    if (in < static_cast<scalar_t>(0)) {
      grad_in_mult_ = -1;
      in = -in;
    } else {
      grad_in_mult_ = 1;
    }
    // `fmod` returns same sign as `in`, which is positive after the `if` above.
    scalar_t extra = ::fmod(in, span);
    int flips = static_cast<int>(::floor(in / span));
    if (flips % 2 == 0) {
      *grad_in = static_cast<scalar_t>(grad_in_mult_);
      return extra + min;
    } else {
      *grad_in = static_cast<scalar_t>(-grad_in_mult_);
      return span - extra + min;
    }
  }
    template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t grid_sampler_unnormalize_set_grad(scalar_t coord, int size,
                                            bool align_corners, scalar_t *grad_in) {
    if (align_corners) {
      // unnormalize coord from [-1, 1] to [0, size - 1]
      *grad_in = static_cast<scalar_t>(size - 1) / 2;
      return ((coord + 1.f) / 2) * (size - 1);
    } else {
      // unnormalize coord from [-1, 1] to [-0.5, size - 0.5]
      *grad_in = static_cast<scalar_t>(size) / 2;
      return ((coord + 1.f) * size - 1) / 2;
    }
  }
  template<typename scalar_t, typename index_t>
  static __forceinline__ __device__
  void safe_add_3d(scalar_t *data, int d, int h, int w,
                  int sD, int sH, int sW, int D, int H, int W,
                  scalar_t delta,
                  const index_t NC_offset) {
    if (within_bounds_3d(d, h, w, D, H, W)) {
      atomicAdd(data+NC_offset + d * sD + h * sH + w * sW,delta);
    }
  }
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t clip_coordinates(scalar_t in, int clip_limit) {
    return ::min(static_cast<scalar_t>(clip_limit - 1), ::max(in, static_cast<scalar_t>(0)));
  }
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
  template <typename scalar_t>
static __forceinline__ __device__
scalar_t clip_coordinates_set_grad(scalar_t in, int clip_limit, scalar_t *grad_in) {
  // Note that it is important for the gradient calculation that borders
  // are considered out of bounds.
  if (in <= static_cast<scalar_t>(0)) {
    *grad_in = static_cast<scalar_t>(0);
    return static_cast<scalar_t>(0);
  } else {
    scalar_t max = static_cast<scalar_t>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<scalar_t>(0);
      return max;
    } else {
      *grad_in = static_cast<scalar_t>(1);
      return in;
    }
  }
}
  template <typename scalar_t>
  static __forceinline__ __device__
  scalar_t grid_sampler_compute_source_index_set_grad(
      scalar_t coord,
      int size,
      int padding_mode,
      bool align_corners,
      scalar_t *grad_in) {
    scalar_t grad_clip, grad_refl;
    coord = grid_sampler_unnormalize_set_grad(coord, size, align_corners, grad_in);
    if (padding_mode == 1) {
      // clip coordinates to image borders
      coord = clip_coordinates_set_grad(coord, size, &grad_clip);
      *grad_in = (*grad_in) * grad_clip;
    } else if (padding_mode == 2) {
      // reflect coordinates by image borders
      if (align_corners) {
        coord = reflect_coordinates_set_grad(coord, 0, 2*(size - 1), &grad_refl);
      } else {
        coord = reflect_coordinates_set_grad(coord, -1, 2*size - 1, &grad_refl);
      }
      // clip coordinates to image borders
      coord = clip_coordinates_set_grad(coord, size, &grad_clip);
      *grad_in = (*grad_in) * grad_refl * grad_clip;
    }

    coord = safe_downgrade_to_int_range(coord);
    return coord;
  }



  template <typename scalar_t, typename index_t>
     __global__ void grid_sampler_3d_backward_kernel(
      const index_t nthreads,
      scalar_t* grad_output,
      scalar_t* input,
      scalar_t* grid,
      scalar_t* grad_input,  // initialized to zeros (or unused if input_requires_grad is false)
      scalar_t* grad_grid,   // initialized to empty
      int*input_sizes,
      int*grid_sizes,
      int*input_strides,
      int*grid_strides,
      int*grad_output_strides,
      int *grad_input_strides,
      const int interpolation_mode,
      const int padding_mode,
      bool align_corners,
      const bool input_requires_grad)
  {
    
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
    
    index_t gOut_sN = grad_output_strides[0];
    index_t gOut_sC = grad_output_strides[1];
    index_t gOut_sD = grad_output_strides[2];
    index_t gOut_sH = grad_output_strides[3];
    index_t gOut_sW = grad_output_strides[4];
    


    // gInp_* (and NC_offset below) are not really needed if input_requires_grad is false.
    int64_t gInp_sN = 0;
    int64_t gInp_sC = 0;
    int64_t gInp_sD = 0;
    int64_t gInp_sH = 0;
    int64_t gInp_sW = 0;
    //if (input_requires_grad) {

      gInp_sN = grad_input_strides[0];
      gInp_sC = grad_input_strides[1];
      gInp_sD = grad_input_strides[2];
      gInp_sH = grad_input_strides[3];
      gInp_sW = grad_input_strides[4];
    //}
    // temp use
    index_t gGrid_sW = grid_sW;

    CUDA_KERNEL_LOOP_TYPE(index, nthreads, index_t) {
      const index_t w = index % out_W;
      const index_t h = (index / out_W) % out_H;
      const index_t d = (index / (out_H * out_W)) % out_D;
      const index_t n = index / (out_D * out_H * out_W);
      const auto grid_offset = n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;

      // get the corresponding input x, y, z co-ordinates from grid
      scalar_t ix = grid[grid_offset];
      scalar_t iy = grid[grid_offset + grid_sCoor];
      scalar_t iz = grid[grid_offset + 2 * grid_sCoor];

      // multipliers for gradients on ix, iy, and iz
      scalar_t gix_mult, giy_mult, giz_mult;
      ix = grid_sampler_compute_source_index_set_grad(ix, inp_W, padding_mode, align_corners, &gix_mult);
      iy = grid_sampler_compute_source_index_set_grad(iy, inp_H, padding_mode, align_corners, &giy_mult);
      iz = grid_sampler_compute_source_index_set_grad(iz, inp_D, padding_mode, align_corners, &giz_mult);

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

        scalar_t gix = static_cast<scalar_t>(0), giy = static_cast<scalar_t>(0), giz = static_cast<scalar_t>(0);
        scalar_t *gOut_ptr_NCDHW = grad_output+ n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
        index_t NC_offset;
        //if (input_requires_grad) {
          NC_offset = n * gInp_sN;
        //}
        scalar_t *inp_ptr_NC = input + n * inp_sN;
        // calculate bilinear weighted pixel value and set output pixel
        for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC, inp_ptr_NC += inp_sC) {
          scalar_t gOut = *gOut_ptr_NCDHW;

          // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
          
          //if (input_requires_grad) {
            safe_add_3d(grad_input, iz_tnw, iy_tnw, ix_tnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tnw * gOut,
                        NC_offset);
            safe_add_3d(grad_input, iz_tne, iy_tne, ix_tne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tne * gOut,
                        NC_offset);
            safe_add_3d(grad_input, iz_tsw, iy_tsw, ix_tsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tsw * gOut,
                        NC_offset);
            safe_add_3d(grad_input, iz_tse, iy_tse, ix_tse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, tse * gOut,
                        NC_offset);
            safe_add_3d(grad_input, iz_bnw, iy_bnw, ix_bnw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bnw * gOut,
                        NC_offset);
            safe_add_3d(grad_input, iz_bne, iy_bne, ix_bne, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bne * gOut,
                        NC_offset);
            safe_add_3d(grad_input, iz_bsw, iy_bsw, ix_bsw, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bsw * gOut,
                        NC_offset);
            safe_add_3d(grad_input, iz_bse, iy_bse, ix_bse, gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, bse * gOut,
                        NC_offset);
         // }
          // calculate grad_grid
          if (within_bounds_3d(iz_tnw, iy_tnw, ix_tnw, inp_D, inp_H, inp_W)) {
            scalar_t tnw_val = inp_ptr_NC[iz_tnw * inp_sD + iy_tnw * inp_sH + ix_tnw * inp_sW];
            gix -= tnw_val * (iy_bse - iy)    * (iz_bse - iz)    * gOut;
            giy -= tnw_val * (ix_bse - ix)    * (iz_bse - iz)    * gOut;
            giz -= tnw_val * (ix_bse - ix)    * (iy_bse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tne, iy_tne, ix_tne, inp_D, inp_H, inp_W)) {
            scalar_t tne_val = inp_ptr_NC[iz_tne * inp_sD + iy_tne * inp_sH + ix_tne * inp_sW];
            gix += tne_val * (iy_bsw - iy)    * (iz_bsw - iz)    * gOut;
            giy -= tne_val * (ix    - ix_bsw) * (iz_bsw - iz)    * gOut;
            giz -= tne_val * (ix    - ix_bsw) * (iy_bsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_tsw, iy_tsw, ix_tsw, inp_D, inp_H, inp_W)) {
            scalar_t tsw_val = inp_ptr_NC[iz_tsw * inp_sD + iy_tsw * inp_sH + ix_tsw * inp_sW];
            gix -= tsw_val * (iy - iy_bne)    * (iz_bne - iz)    * gOut;
            giy += tsw_val * (ix_bne - ix)    * (iz_bne - iz)    * gOut;
            giz -= tsw_val * (ix_bne - ix)    * (iy    - iy_bne) * gOut;
          }
          if (within_bounds_3d(iz_tse, iy_tse, ix_tse, inp_D, inp_H, inp_W)) {
            scalar_t tse_val = inp_ptr_NC[iz_tse * inp_sD + iy_tse * inp_sH + ix_tse * inp_sW];
            gix += tse_val * (iy - iy_bnw)    * (iz_bnw - iz)    * gOut;
            giy += tse_val * (ix    - ix_bnw) * (iz_bnw - iz)    * gOut;
            giz -= tse_val * (ix    - ix_bnw) * (iy    - iy_bnw) * gOut;
          }
          if (within_bounds_3d(iz_bnw, iy_bnw, ix_bnw, inp_D, inp_H, inp_W)) {
            scalar_t bnw_val = inp_ptr_NC[iz_bnw * inp_sD + iy_bnw * inp_sH + ix_bnw * inp_sW];
            gix -= bnw_val * (iy_tse - iy)    * (iz - iz_tse)    * gOut;
            giy -= bnw_val * (ix_tse - ix)    * (iz - iz_tse)    * gOut;
            giz += bnw_val * (ix_tse - ix)    * (iy_tse - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bne, iy_bne, ix_bne, inp_D, inp_H, inp_W)) {
            scalar_t bne_val = inp_ptr_NC[iz_bne * inp_sD + iy_bne * inp_sH + ix_bne * inp_sW];
            gix += bne_val * (iy_tsw - iy)    * (iz - iz_tsw)    * gOut;
            giy -= bne_val * (ix    - ix_tsw) * (iz - iz_tsw)    * gOut;
            giz += bne_val * (ix    - ix_tsw) * (iy_tsw - iy)    * gOut;
          }
          if (within_bounds_3d(iz_bsw, iy_bsw, ix_bsw, inp_D, inp_H, inp_W)) {
            scalar_t bsw_val = inp_ptr_NC[iz_bsw * inp_sD + iy_bsw * inp_sH + ix_bsw * inp_sW];
            gix -= bsw_val * (iy - iy_tne)    * (iz - iz_tne)    * gOut;
            giy += bsw_val * (ix_tne - ix)    * (iz - iz_tne)    * gOut;
            giz += bsw_val * (ix_tne - ix)    * (iy    - iy_tne) * gOut;
          }
          if (within_bounds_3d(iz_bse, iy_bse, ix_bse, inp_D, inp_H, inp_W)) {
            scalar_t bse_val = inp_ptr_NC[iz_bse * inp_sD + iy_bse * inp_sH + ix_bse * inp_sW];
            gix += bse_val * (iy - iy_tnw)    * (iz - iz_tnw)    * gOut;
            giy += bse_val * (ix    - ix_tnw) * (iz - iz_tnw)    * gOut;
            giz += bse_val * (ix    - ix_tnw) * (iy    - iy_tnw) * gOut;
          }
        }

        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = gix_mult * gix;
        gGrid_ptr_NDHW[1] = giy_mult * giy;
        gGrid_ptr_NDHW[2] = giz_mult * giz;
      } else if (interpolation_mode == 1) {
       
        if (input_requires_grad) {
          auto ix_nearest = static_cast<index_t>(::round(ix));
          auto iy_nearest = static_cast<index_t>(::round(iy));
          auto iz_nearest = static_cast<index_t>(::round(iz));

          // assign nearest neighor pixel value to output pixel
          scalar_t *gOut_ptr_NCDHW = grad_output+ n * gOut_sN + d * gOut_sD + h * gOut_sH + w * gOut_sW;
          index_t NC_offset = n * gInp_sN;
          for (index_t c = 0; c < C; ++c, gOut_ptr_NCDHW += gOut_sC, NC_offset += gInp_sC) {
            // calculate and set grad_input. See Note [Passing pointer and offset to fastAtomicAdd].
            safe_add_3d(grad_input, iz_nearest, iy_nearest, ix_nearest,
                        gInp_sD, gInp_sH, gInp_sW, inp_D, inp_H, inp_W, *gOut_ptr_NCDHW,
                        NC_offset);
          }
        }
        // assuming grad_grid is contiguous
        // thus we can
        //   1. use index with gGrid_sW to directly compute gGrid_ptr_NDHW
        //   2. directly assign to gGrid_ptr_NDHW[0], gGrid_ptr_NDHW[1], gGrid_ptr_NDHW[2]
        scalar_t *gGrid_ptr_NDHW = grad_grid + index * gGrid_sW;
        gGrid_ptr_NDHW[0] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[1] = static_cast<scalar_t>(0);
        gGrid_ptr_NDHW[2] = static_cast<scalar_t>(0);
      }
    }
  }
}
    ''',
    cuda_src=f'''
    @alias(grad_output, in0)
    @alias(input, in1)
    @alias(grid, in2)
    @alias(grad_input, out0)
    @alias(grad_grid, out1)
    
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
  

    
    index_t gOut_sW = 1;
    index_t gOut_sH = gOut_sW*grad_output_shape4;
    index_t gOut_sD = gOut_sH*grad_output_shape3;
    index_t gOut_sC = gOut_sD*grad_output_shape2;
    index_t gOut_sN = gOut_sC*grad_output_shape1;
    thrust::host_vector <int > host_grad_output_strides (5);

    host_grad_output_strides[0]=gOut_sN;
    host_grad_output_strides[1]=gOut_sC;
    host_grad_output_strides[2]=gOut_sD;
    host_grad_output_strides[3]=gOut_sH;
    host_grad_output_strides[4]=gOut_sW;
    thrust::device_vector <int > grad_output_strides=host_grad_output_strides;
    int*grad_output_strides_ptr=thrust::raw_pointer_cast(&grad_output_strides[0]);

    int gInp_sW=1;
    int gInp_sH=grad_input_shape4*gInp_sW;
    int gInp_sD=grad_input_shape3*gInp_sH;
    int gInp_sC=grad_input_shape2*gInp_sD;
    int gInp_sN=grad_input_shape1*gInp_sC;
    thrust :: host_vector <int > host_grad_input_strides (5);
    host_grad_input_strides[0]=gInp_sN;
    host_grad_input_strides[1]=gInp_sC;
    host_grad_input_strides[2]=gInp_sD;
    host_grad_input_strides[3]=gInp_sH;
    host_grad_input_strides[4]=gInp_sW;
    thrust :: device_vector <int > grad_input_strides=host_grad_input_strides;

    int*grad_input_strides_ptr=thrust::raw_pointer_cast(&grad_input_strides[0]);


    auto N = input_shape0;
    auto D = grid_shape1;
    auto H = grid_shape2;
    auto W = grid_shape3;
    int64_t count = N * D * H * W;
    if (count>0)
    {{
        const int num_threads = 512;
        const int blocks = (count + num_threads - 1 ) / num_threads;
        grid_sampler_3d_backward_kernel<float32><<<blocks, num_threads>>>(
        count,
      grad_output_p,
      input_p,
      grid_p,
      grad_input_p,  // initialized to zeros (or unused if input_requires_grad is false)
      grad_grid_p,   // initialized to empty
      input_sizes_ptr,
      grid_sizes_ptr,
      input_strides_ptr,
      grid_strides_ptr,
      grad_output_strides_ptr,
      grad_input_strides_ptr,
     {interpolation_mode},
      {padding_mode},
      {align_corners},1
      );
    }}
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in grid sample3d backward: %s\\n", cudaGetErrorString(err));
        
    ''')
    return grad_input,grad_grid


if __name__ == '__main__':
    import numpy as np 
    jt.flags.use_cuda = 2

    #grid = np.random.rand(1,1,89,100,120).astype("float32").clip(0.1,0.8)
    grid = np.random.randn(1,3,99,101,101)
    xyz = np.random.randn(629236,3)
    xyz_min = np.min(xyz,axis=0)
    xyz_max = np.max(xyz,axis=0)
    nor_xyz = np.flip(((xyz - xyz_min) / (xyz_max - xyz_min)),-1) * 2 - 1
    ind_norm = nor_xyz.reshape(1,1,1,-1,3)

    # grid=np.load("grid.npy")
    # ind_norm=np.load("ind_norm.npy")

    ind_norm_jittor = jt.array(ind_norm)
    grid_jittor  = jt.array(grid)
    # output=np.load("grid_sample_out.npy")
    

    
    import torch
    import  torch.nn.functional as F
    import random
    # torch.manual_seed(777)
    # np.random.seed(777)
    # random.seed(777)
    # torch.backends.cudnn.enabled = False
    grid_torch   = torch.from_numpy(grid).cuda(device='cuda:0').requires_grad_(True)
    ind_norm_torch  = torch.from_numpy(ind_norm).cuda(device='cuda:0').requires_grad_(True)

    print(grid_torch.shape)
    print(ind_norm_torch.shape)
    out = F.grid_sample(grid_torch,ind_norm_torch, mode='bilinear', align_corners=True)

    out_grad=torch.rand_like(out).cuda(device='cuda:0')
    out.backward(out_grad,retain_graph=True)

    grad_grid=grid_torch.grad.cpu().numpy()
    output_jittor= jt.array(out.detach().cpu().numpy())
    jt_input=jt.array(out_grad.cpu().numpy())
    import time
    start=time.time()
    for i in range(1000):
        grad_input,grad_grid=grid_sampler_3d_backward_cuda(jt_input,grid_jittor,ind_norm_jittor,0,0,1)
        jt.sync_all()
    print("time:",(time.time()-start)/1000.0)
    diff=grad_input.data-grid_torch.grad.cpu().numpy()
    diff2=grad_grid.data-ind_norm_torch.grad.cpu().numpy()
    # print(np.sum(grad_input.data!=0))
    # print(np.sum(grad_grid.data!=0))
    # print(np.sum(grid_torch.grad.cpu().numpy()!=0))
    # print(np.sum(ind_norm_torch.grad.cpu().numpy()!=0))
    
    print(np.max(np.abs(diff)))
    # print(np.sum(np.abs(diff2)>1e-5))
   
    # print(np.max(np.abs(diff)))
    # print(np.max(np.abs(diff)))
    # print(grad_input.data[index].flatten()[0:100])
    # print(grid_torch.grad.cpu().numpy()[index].flatten()[0:100])
