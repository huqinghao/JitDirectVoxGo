import jittor as jt
   
def adam_upd(
    param, 
    grad, 
    exp_avg, 
    exp_avg_sq, 
    step, 
    beta1, 
    beta2, 
    lr, 
    eps):
    return jt.code([],[],[param, grad, exp_avg, exp_avg_sq],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{
template <typename scalar_t>
__global__ void adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    const float step_size, const float beta1, const float beta2, const float eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}
}
    ''',
    
    cuda_src=f'''
    @alias(param, in0)
    @alias(grad, in1)
    @alias(exp_avg, in2)
    @alias(exp_avg_sq, in3)
    
    const size_t N = in0->numel()

    const int threads = 512;
    const int blocks = (N + threads - 1) / threads;

    const float step_size = {lr} * sqrt(1 - pow({beta2}, (float){step})) / (1 - pow({beta1}, (float){step}));

    /*
    AT_DISPATCH_FLOATING_TYPES(param.type(), "adam_upd_cuda", ([&] {
        adam_upd_cuda_kernel<scalar_t><<<blocks, threads>>>(
            param.data<scalar_t>(),
            grad.data<scalar_t>(),
            exp_avg.data<scalar_t>(),
            exp_avg_sq.data<scalar_t>(),
            N, step_size, beta1, beta2, eps);
    }));
    */

    adam_upd_cuda_kernel<float32><<<blocks, threads>>>(
        param_p,
        grad_p,
        exp_avg_p,
        exp_avg_sq_p,
        N, step_size, {beta1}, {beta2}, {eps});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in adam_upd: %s\\n", cudaGetErrorString(err));
    ''')


def masked_adam_upd_cuda_kernel(
    param, 
    grad, 
    exp_avg, 
    exp_avg_sq, 
    step, 
    beta1, 
    beta2, 
    lr, 
    eps):
    return jt.code([],[],[param, grad, exp_avg, exp_avg_sq],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{
template <typename scalar_t>
__global__ void masked_adam_upd_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    const size_t N,
    const float step_size, const float beta1, const float beta2, const float eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N && grad[index]!=0) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}
}
    ''',
    
    cuda_src=f'''
    @alias(param, in0)
    @alias(grad, in1)
    @alias(exp_avg, in2)
    @alias(exp_avg_sq, in3)
    
    const size_t N = in0->numel()

    const int threads = 512;
    const int blocks = (N + threads - 1) / threads;

    const float step_size = {lr} * sqrt(1 - pow({beta2}, (float){step})) / (1 - pow({beta1}, (float){step}));

    /*
    AT_DISPATCH_FLOATING_TYPES(param.type(), "masked_adam_upd_cuda", ([&] {
        masked_adam_upd_cuda_kernel<scalar_t><<<blocks, threads>>>(
            param.data<scalar_t>(),
            grad.data<scalar_t>(),
            exp_avg.data<scalar_t>(),
            exp_avg_sq.data<scalar_t>(),
            N, step_size, beta1, beta2, eps);
    }));
    */

    masked_adam_upd_cuda_kernel<float32><<<blocks, threads>>>(
        param_p,
        grad_p,
        exp_avg_p,
        exp_avg_sq_p,
        N, step_size, {beta1}, {beta2}, {eps});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in masked_adam_upd_cuda_kernel: %s\\n", cudaGetErrorString(err));
    ''')


def adam_upd_with_perlr(
    param, 
    grad, 
    exp_avg, 
    exp_avg_sq, 
    perlr, 
    step, 
    beta1, 
    beta2, 
    lr, 
    eps):
    return jt.code([],[],[param, grad, exp_avg, exp_avg_sq, perlr],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{
template <typename scalar_t>
__global__ void adam_upd_with_perlr_cuda_kernel(
    scalar_t* __restrict__ param,
    const scalar_t* __restrict__ grad,
    scalar_t* __restrict__ exp_avg,
    scalar_t* __restrict__ exp_avg_sq,
    scalar_t* __restrict__ perlr,
    const size_t N,
    const float step_size, const float beta1, const float beta2, const float eps) {

  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if(index<N) {
    exp_avg[index] = beta1 * exp_avg[index] + (1-beta1) * grad[index];
    exp_avg_sq[index] = beta2 * exp_avg_sq[index] + (1-beta2) * grad[index] * grad[index];
    param[index] -= step_size * perlr[index] * exp_avg[index] / (sqrt(exp_avg_sq[index]) + eps);
  }
}
}
    ''',
    cuda_src=f'''
    @alias(param, in0)
    @alias(grad, in1)
    @alias(exp_avg, in2)
    @alias(exp_avg_sq, in3)
    @alias(perlr, in4)
    
    const size_t N = in0->numel()

    const int threads = 512;
    const int blocks = (N + threads - 1) / threads;

    const float step_size = {lr} * sqrt(1 - pow({beta2}, (float){step})) / (1 - pow({beta1}, (float){step}));

    /*
    AT_DISPATCH_FLOATING_TYPES(param.type(), "adam_upd_with_perlr_cuda", ([&] {
        adam_upd_with_perlr_cuda_kernel<scalar_t><<<blocks, threads>>>(
            param.data<scalar_t>(),
            grad.data<scalar_t>(),
            exp_avg.data<scalar_t>(),
            exp_avg_sq.data<scalar_t>(),
            perlr.data<scalar_t>(),
            N, step_size, beta1, beta2, eps);
    }));
    */
    
    adam_upd_with_perlr_cuda_kernel<float32><<<blocks, threads>>>(
        param_p,
        grad_p,
        exp_avg_p,
        exp_avg_sq_p,
        perlr_p,
        N, step_size, {beta1}, {beta2}, {eps});
    
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in adam_upd_with_perlr: %s\\n", cudaGetErrorString(err));
    ''')
    