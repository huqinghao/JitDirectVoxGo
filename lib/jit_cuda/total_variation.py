import jittor as jt

def total_variation_add_grad(
    param, 
    grad, 
    wx, 
    wy, 
    wz, 
    dense_mode):
    return jt.code([],[],[param, grad],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{
template <typename scalar_t, typename bound_t>
__device__ __forceinline__ scalar_t clamp(const scalar_t v, const bound_t lo, const bound_t hi) {
  return min(max(v, lo), hi);
}

template <typename scalar_t, bool dense_mode>
__global__ void total_variation_add_grad_cuda_kernel(
    const scalar_t* __restrict__ param,
    scalar_t* __restrict__ grad,
    float wx, float wy, float wz,
    const size_t sz_i, const size_t sz_j, const size_t sz_k, const size_t N) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N && (dense_mode || grad[index]!=0)) {
    const size_t k = index % sz_k;
    const size_t j = index / sz_k % sz_j;
    const size_t i = index / sz_k / sz_j % sz_i;

    float grad_to_add = 0;
    grad_to_add += (k==0      ? 0 : wz * clamp(param[index]-param[index-1], -1.f, 1.f));
    grad_to_add += (k==sz_k-1 ? 0 : wz * clamp(param[index]-param[index+1], -1.f, 1.f));
    grad_to_add += (j==0      ? 0 : wy * clamp(param[index]-param[index-sz_k], -1.f, 1.f));
    grad_to_add += (j==sz_j-1 ? 0 : wy * clamp(param[index]-param[index+sz_k], -1.f, 1.f));
    grad_to_add += (i==0      ? 0 : wz * clamp(param[index]-param[index-sz_k*sz_j], -1.f, 1.f));
    grad_to_add += (i==sz_i-1 ? 0 : wz * clamp(param[index]-param[index+sz_k*sz_j], -1.f, 1.f));
    grad[index] += grad_to_add;
  }
}
}
    ''',
    cuda_src=f'''
    @alias(param, in0)
    @alias(grad, in1)
    
    const size_t N = in0->numel()
    const size_t sz_i = param_shape2;
    const size_t sz_j = param_shape3;
    const size_t sz_k = param_shape4;
    
    const int threads = 512;
    const int blocks = (N + threads - 1) / threads;

    wx_6 = {wx} / 6;
    wy_6 = {wy} / 6;
    wz_6 = {wz} / 6;

    if({dense_mode}) {
        total_variation_add_grad_cuda_kernel<float32,true><<<blocks, threads>>>(
            param_p,
            grad_p,
            wx_6, wy_6, wz_6,
            sz_i, sz_j, sz_k, N);          
    }
    else {
        total_variation_add_grad_cuda_kernel<float32,false><<<blocks, threads>>>(
            param_p,
            grad_p,
            wx_6, wy_6, wz_6,
            sz_i, sz_j, sz_k, N);    
    }


    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in total_variation_add_grad: %s\\n", cudaGetErrorString(err));
    ''')

