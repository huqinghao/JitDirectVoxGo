import jittor as jt

def infer_t_minmax(
    rays_o,
    rays_d,
    xyz_min,
    xyz_max,
    near,
    far,
    ):
    
    t_min,t_max=jt.code([(rays_o.size(0),),(rays_o.size(0),)],[rays_o.dtype,rays_o.dtype],[rays_o,rays_d,xyz_min,xyz_max],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*
   Points sampling helper functions.
 */
 
namespace{
template <typename scalar_t>
__global__ void infer_t_minmax_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        const float near, const float far, const int n_rays,
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    float vx = ((rays_d[offset  ]==0) ? 1e-6 : rays_d[offset  ]);
    float vy = ((rays_d[offset+1]==0) ? 1e-6 : rays_d[offset+1]);
    float vz = ((rays_d[offset+2]==0) ? 1e-6 : rays_d[offset+2]);
    float ax = (xyz_max[0] - rays_o[offset  ]) / vx;
    float ay = (xyz_max[1] - rays_o[offset+1]) / vy;
    float az = (xyz_max[2] - rays_o[offset+2]) / vz;
    float bx = (xyz_min[0] - rays_o[offset  ]) / vx;
    float by = (xyz_min[1] - rays_o[offset+1]) / vy;
    float bz = (xyz_min[2] - rays_o[offset+2]) / vz;
    t_min[i_ray] = max(min(max(max(min(ax, bx), min(ay, by)), min(az, bz)), far), near);
    t_max[i_ray] = max(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), far), near);
  }
}
}
    ''',
    cuda_src=f'''
    @alias(rays_o, in0)
    @alias(rays_d, in1)
    @alias(xyz_min, in2)
    @alias(xyz_max, in3)
    @alias(t_min, out0)
    @alias(t_max, out1)
    
    cudaMemsetAsync(out0_p, 0, out0->size);
    cudaMemsetAsync(out1_p, 0, out1->size);
    
    const int n_rays = rays_o_shape0;
    
    const int threads = 512;
    const int blocks = (n_rays + threads - 1) / threads;
    
    
    infer_t_minmax_cuda_kernel<float32><<<blocks, threads>>>(
        rays_o_p,
        rays_d_p,
        xyz_min_p,
        xyz_max_p,
        {near}, {far}, n_rays,
        t_min_p,
        t_max_p);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in infer_t_minmax: %s\\n", cudaGetErrorString(err));
    ''')
    return t_min,t_max
    
def infer_n_samples(
    rays_d,
    t_min,
    t_max,
    stepdist):
    return jt.code((t_min.size(0),),jt.int64,[rays_d, t_min, t_max],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using jittor::int64;

namespace{

template <typename scalar_t>
__global__ void infer_n_samples_cuda_kernel(
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max,
        const float stepdist,
        const int n_rays,
        int64* __restrict__ n_samples) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    const float rnorm = sqrt(
            rays_d[offset  ]*rays_d[offset  ] +\
            rays_d[offset+1]*rays_d[offset+1] +\
            rays_d[offset+2]*rays_d[offset+2]);
    // at least 1 point for easier implementation in the later sample_pts_on_rays_cuda
    n_samples[i_ray] = max(ceil((t_max[i_ray]-t_min[i_ray]) * rnorm / stepdist), 1.);
  }
}
}

    ''',
    cuda_src=f'''
    @alias(rays_d, in0)
    @alias(t_min, in1)
    @alias(t_max, in2)
    @alias(n_samples, out0)
    
    const int n_rays = t_min_shape0;
    
    cudaMemsetAsync(out0_p, 0, out0->size);
    const int threads = 512;
    const int blocks = (n_rays + threads - 1) / threads;
    const float step_dist = {stepdist};

    
    infer_n_samples_cuda_kernel<float32><<<blocks, threads>>>(
        rays_d_p,
        t_min_p,
        t_max_p,
        step_dist,
        n_rays,
        n_samples_p);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in infer_n_samples: %s\\n", cudaGetErrorString(err));
    
    ''')



def infer_ray_start_dir(
    rays_o,
    rays_d,
    t_min):
    return jt.code([rays_o.shape,rays_o.shape],[rays_o.dtype,rays_o.dtype],[rays_o,rays_d,t_min],
    cuda_header='''
    
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{
    
template <typename scalar_t>
__global__ void infer_ray_start_dir_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ t_min,
        const int n_rays,
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    const float rnorm = sqrt(
            rays_d[offset  ]*rays_d[offset  ] +\
            rays_d[offset+1]*rays_d[offset+1] +\
            rays_d[offset+2]*rays_d[offset+2]);
    rays_start[offset  ] = rays_o[offset  ] + rays_d[offset  ] * t_min[i_ray];
    rays_start[offset+1] = rays_o[offset+1] + rays_d[offset+1] * t_min[i_ray];
    rays_start[offset+2] = rays_o[offset+2] + rays_d[offset+2] * t_min[i_ray];
    rays_dir  [offset  ] = rays_d[offset  ] / rnorm;
    rays_dir  [offset+1] = rays_d[offset+1] / rnorm;
    rays_dir  [offset+2] = rays_d[offset+2] / rnorm;
  }
}
    
    
}
    
    ''',
    cuda_src=f'''
    @alias(rays_o, in0)
    @alias(rays_d, in1)
    @alias(t_min, in2)
    @alias(rays_start, out0)
    @alias(rays_dir, out1)

    const int n_rays = rays_o_shape0;
    const int threads = 512;
    const int blocks = (n_rays + threads - 1) / threads;

    cudaMemsetAsync(out0_p, 0, out0->size);
    cudaMemsetAsync(out1_p, 0, out1->size);

 
    infer_ray_start_dir_cuda_kernel<float32><<<blocks, threads>>>(
        rays_o_p,
        rays_d_p,
        t_min_p,
        n_rays,
        rays_start_p,
        rays_dir_p);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in infer_ray_start_dir: %s\\n", cudaGetErrorString(err));
    ''')
    
    

def sample_pts_on_rays(rays_o, rays_d, xyz_min, xyz_max, near, far, stepdist):
    
    assert(len(rays_o.shape)==2)
    assert(rays_o.size(1)==3)
    
    threads = 512
    n_rays = rays_o.size(0)
    
    t_min,t_max = infer_t_minmax(rays_o, rays_d, xyz_min, xyz_max, near, far)
    
    N_steps = infer_n_samples(rays_d, t_min, t_max, stepdist)
    
    # N_steps_cumsum = N_steps.cumsum(0) # 211 TODO check
    
    N_steps_cumsum = jt.cumsum(N_steps,0) # 211 TODO check
    
    total_len = int(N_steps.data.sum().item()) # 212 TODO check
    
    ray_id = jt.zeros((total_len),dtype="int64")
    
    # __set_1_at_ray_seg_start
    # jt.code((1,),"float32", [ray_id,N_steps_cumsum], cuda_header='''
    jt.code([],[ray_id,N_steps_cumsum], cuda_header='''
            
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using jittor::int64;
namespace{     

__global__ void __set_1_at_ray_seg_start(
        int64* __restrict__ ray_id,
        int64* __restrict__ N_steps_cumsum,
        const int n_rays) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<idx && idx<n_rays) {
    ray_id[N_steps_cumsum[idx-1]] = 1;
  }
}           

}
    ''', 
    cuda_src=f'''
    @alias(ray_id, out0)
    @alias(N_steps_cumsum, out1)
    
    __set_1_at_ray_seg_start<<<({n_rays}+{threads}-1)/{threads}, {threads}>>>(
      ray_id_p, N_steps_cumsum_p, {n_rays});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in __set_1_at_ray_seg_start: %s\\n", cudaGetErrorString(err));
    
    ''')
    
    ray_id = jt.cumsum(ray_id,0)

    step_id = jt.empty((total_len,),dtype=ray_id.dtype)
    
    # __set_step_id
    jt.code((1,),"float32", [step_id, ray_id, N_steps_cumsum], cuda_header='''
            
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using jittor::int64

namespace{     

__global__ void __set_step_id(
        int64* __restrict__ step_id,
        int64* __restrict__ ray_id,
        int64* __restrict__ N_steps_cumsum,
        const int total_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<total_len) {
      const int rid = ray_id[idx];
      step_id[idx] = idx - ((rid!=0) ? N_steps_cumsum[rid-1] : 0);
    }
}         

}            
    ''', 
    cuda_src=f'''
    @alias(step_id, in0)
    @alias(ray_id, in1)
    @alias(N_steps_cumsum, in2)
    @alias(place_holder, out0)

    cudaMemsetAsync(out0_p, 0, out0->size);
    
    __set_step_id<<<({total_len}+{threads}-1)/{threads}, {threads}>>>(
      step_id_p, ray_id_p, N_steps_cumsum_p, {total_len});

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in __set_step_id: %s\\n", cudaGetErrorString(err));
    
    ''')
    
    rays_start,rays_dir = infer_ray_start_dir(rays_o, rays_d, t_min)
    
    rays_pts = jt.empty((total_len,3),rays_o.dtype)
    
    mask_outbbox = jt.empty((total_len,), dtype='bool')
    
    
    jt.code((1,),"float32",[rays_start, rays_dir, xyz_min, xyz_max, ray_id, step_id, rays_pts, mask_outbbox],
    cuda_header='''
    
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

/*
   Sampling query points on rays.
 */
 
template <typename scalar_t>
__global__ void sample_pts_on_rays_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        int64* __restrict__ ray_id,
        int64* __restrict__ step_id,
        const float stepdist, const int total_len,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ mask_outbbox) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = stepdist * i_step;
    const float px = rays_start[offset_r  ] + rays_dir[offset_r  ] * dist;
    const float py = rays_start[offset_r+1] + rays_dir[offset_r+1] * dist;
    const float pz = rays_start[offset_r+2] + rays_dir[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    mask_outbbox[idx] = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
  }
}


}  
    
    ''',
    cuda_src=f''' 

    @alias(rays_start, in0)
    @alias(rays_dir, in1)
    @alias(xyz_min, in2)
    @alias(xyz_max, in3)
    @alias(ray_id, in4)
    @alias(step_id, in5)
    @alias(rays_pts, in6)
    @alias(mask_outbbox, in7)
    @alias(place_holder, out0)

    cudaMemsetAsync(out0_p, 0, out0->size);

    sample_pts_on_rays_cuda_kernel<float32><<<({total_len}+{threads}-1)/{threads}, {threads}>>>(
        rays_start_p,
        rays_dir_p,
        xyz_min_p,
        xyz_max_p,
        ray_id_p,
        step_id_p,
        {stepdist}, {total_len},
        rays_pts_p,
        mask_outbbox_p);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in sample_pts_on_rays: %s\\n", cudaGetErrorString(err));
    ''')
    
    return rays_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max
    
def maskcache_lookup(world, xyz, xyz2ijk_scale, xyz2ijk_shift):
    assert(world.ndim==3)
    assert(xyz.ndim==2)
    assert(xyz.size(1)==3)
    
    return jt.code((xyz.size(0),),world.dtype,[world,xyz,xyz2ijk_scale, xyz2ijk_shift],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{


/*
   MaskCache lookup to skip known freespace.
 */

static __forceinline__ __device__
bool check_xyz(int i, int j, int k, int sz_i, int sz_j, int sz_k) {
  return (0 <= i) && (i < sz_i) && (0 <= j) && (j < sz_j) && (0 <= k) && (k < sz_k);
}


template <typename scalar_t>
__global__ void maskcache_lookup_cuda_kernel(
    bool* __restrict__ world,
    scalar_t* __restrict__ xyz,
    bool* __restrict__ out,
    scalar_t* __restrict__ xyz2ijk_scale,
    scalar_t* __restrict__ xyz2ijk_shift,
    const int sz_i, const int sz_j, const int sz_k, const int n_pts) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const int offset = i_pt * 3;
    const int i = round(xyz[offset  ] * xyz2ijk_scale[0] + xyz2ijk_shift[0]);
    const int j = round(xyz[offset+1] * xyz2ijk_scale[1] + xyz2ijk_shift[1]);
    const int k = round(xyz[offset+2] * xyz2ijk_scale[2] + xyz2ijk_shift[2]);
    if(check_xyz(i, j, k, sz_i, sz_j, sz_k)) {
      out[i_pt] = world[i*sz_j*sz_k + j*sz_k + k];
    }
  }
}



}  
    
    
    ''',
    cuda_src=f'''
    @alias(world, in0)
    @alias(xyz, in1)
    @alias(xyz2ijk_scale, in2)
    @alias(xyz2ijk_shift, in3)
    @alias(maskcache, out0)
    
    cudaMemsetAsync(out0_p, 0, out0->size);

    const int sz_i = world_shape0;
    const int sz_j = world_shape1;
    const int sz_k = world_shape2;
    const int n_pts = xyz_shape0;
    
    if(n_pts!=0) {{
      const int threads = 512;
      const int blocks = (n_pts + threads - 1) / threads;


      
      
      maskcache_lookup_cuda_kernel<float32><<<blocks, threads>>>(
        world_p,
        xyz_p,
        maskcache_p,
        xyz2ijk_scale_p,
        xyz2ijk_shift_p,
        sz_i, sz_j, sz_k, n_pts);
        
    }}    
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in maskcache_lookup: %s\\n", cudaGetErrorString(err));
    ''')
    
    
    
def raw2alpha(
    density, 
    shift, 
    interval):
    assert(density.ndim==1)
    return jt.code([density.shape,density.shape],[density.dtype,density.dtype],[density],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

/*
    Ray marching helper function.
 */
template <typename scalar_t>
__global__ void raw2alpha_cuda_kernel(
    scalar_t* __restrict__ density,
    const float shift, const float interval, const int n_pts,
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ alpha) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const scalar_t e = exp(density[i_pt] + shift); // can be inf
    exp_d[i_pt] = e;
    alpha[i_pt] = 1 - pow(1 + e, -interval);
  }
}

}  
    
  ''',
  cuda_src=f'''
  @alias(density, in0)
  @alias(exp_d, out0)
  @alias(alpha, out1)
  
  const int n_pts = density_shape0;
  cudaMemsetAsync(out0_p, 0, out0->size);
  cudaMemsetAsync(out1_p, 0, out1->size);
  
  if(n_pts!=0) {{ 
    const int threads = 512;
    const int blocks = (n_pts + threads - 1) / threads;
    

    
    raw2alpha_cuda_kernel<float32><<<blocks, threads>>>(
      density_p,
      {shift}, {interval}, n_pts,
      exp_d_p,
      alpha_p);
    
  }}
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
          printf("Error in raw2alpha: %s\\n", cudaGetErrorString(err));
  ''')
    
    

def raw2alpha_backward(
    exp_d, 
    grad_back, 
    interval):
    return jt.code(exp_d.shape,exp_d.dtype,[exp_d,grad_back],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

template <typename scalar_t>
__global__ void raw2alpha_backward_cuda_kernel(
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ grad_back,
    const float interval, const int n_pts,
    scalar_t* __restrict__ grad) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    grad[i_pt] = min(exp_d[i_pt], 1e10) * pow(1+exp_d[i_pt], -interval-1) * interval * grad_back[i_pt];
  }
}

}  
    
    ''',
    cuda_src=f'''
    @alias(exp_d, in0)
    @alias(grad_back, in1)
    @alias(grad, out0)

    
    const int n_pts = exp_d_shape0;
    cudaMemsetAsync(out0_p, 0, out0->size);
    
    if(n_pts!=0) {{
        const int threads = 512;
        const int blocks = (n_pts + threads - 1) / threads;
        
   

        raw2alpha_backward_cuda_kernel<float32><<<blocks, threads>>>(
          exp_d_p,
          grad_back_p,
          {interval}, n_pts,
          grad_p);
    }}
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in raw2alpha_backward: %s\\n", cudaGetErrorString(err));
    ''')
    
    
    
def raw2alpha_nonuni(density, shift, interval):
    assert(density.ndim==1)
    return jt.code([density.shape,density.shape],[density.dtype,density.dtype],[density,interval],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

template <typename scalar_t>
__global__ void raw2alpha_nonuni_cuda_kernel(
    scalar_t* __restrict__ density,
    const float shift, scalar_t* __restrict__ interval, const int n_pts,
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ alpha) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const scalar_t e = exp(density[i_pt] + shift); // can be inf
    exp_d[i_pt] = e;
    alpha[i_pt] = 1 - pow(1 + e, -interval[i_pt]);
  }
}

}  
    
    
    ''',
    cuda_src=f'''
    @alias(density, in0)
    @alias(interval, in1)
    @alias(exp_d, out0)
    @alias(alpha, out1)
    
    const int n_pts = density_shape0; 
    cudaMemsetAsync(out0_p, 0, out0->size);
    cudaMemsetAsync(out1_p, 0, out1->size);

    if(n_pts != 0) {{
        const int threads = 256;
        const int blocks = (n_pts + threads - 1) / threads;
        raw2alpha_nonuni_cuda_kernel<float32><<<blocks, threads>>>(
            density_p,
            {shift}, interval_p, n_pts,
            exp_d_p,
            alpha_p);
    }}  
    
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in raw2alpha_nonuni: %s\\n", cudaGetErrorString(err));
    ''')
    
    
    

def raw2alpha_nonuni_backward(exp_d, grad_back, interval):
    return jt.code(exp_d.shape,exp_d.dtype,[exp_d, grad_back, interval],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

template <typename scalar_t>
__global__ void raw2alpha_nonuni_backward_cuda_kernel(
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ grad_back,
    scalar_t* __restrict__ interval, const int n_pts,
    scalar_t* __restrict__ grad) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    grad[i_pt] = min(exp_d[i_pt], 1e10) * pow(1+exp_d[i_pt], -interval[i_pt]-1) * interval[i_pt] * grad_back[i_pt];
  }
}

}  
    
    
    ''',
    cuda_src=f'''   
    @alias(exp_d, in0)
    @alias(grad_back, in1)
    @alias(interval, in2)
    @alias(grad, out0)
    
    const int n_pts = exp_d_shape0;
    cudaMemsetAsync(out0_p, 0, out0->size);

    if(n_pts != 0) {{
        const int threads = 256;
        const int blocks = (n_pts + threads - 1) / threads;
        raw2alpha_nonuni_backward_cuda_kernel<float32><<<blocks, threads>>>(
          exp_d_p,
          grad_back_p,
          interval_p, n_pts,
          grad_p);
        
    }}
        
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in raw2alpha_nonuni_backward: %s\\n", cudaGetErrorString(err));
    ''')
    
    
    
def alpha2weight(alpha, ray_id, n_rays):
    assert(alpha.ndim==1)
    assert(ray_id.ndim==1)
    assert(alpha.numel()==ray_id.numel())
    
    n_pts = alpha.size(0)
    threads = 512
    
    weight = jt.zeros_like(alpha)
    T = jt.ones_like(alpha)
    alphainv_last = jt.ones((n_rays),alpha.dtype)
    i_start = jt.zeros((n_rays),'int64')
    i_end = jt.zeros((n_rays),'int64')
    
    if (n_pts== 0):
          return weight,T,alphainv_last,i_start,i_end
    
    jt.code(inputs=[ray_id,i_start,i_end],outputs=[ray_id,i_start,i_end],cuda_header='''
            
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using  jittor::int64;
namespace{

__global__ void __set_i_for_segment_start_end(
    int64* __restrict__ ray_id,
    const int n_pts,
    int64* __restrict__ i_start,
    int64* __restrict__ i_end) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<index && index<n_pts && ray_id[index]!=ray_id[index-1]) {
    i_start[ray_id[index]] = index;
    i_end[ray_id[index-1]] = index;
  }
}

}     

    ''', 
    cuda_src=f'''
    @alias(ray_id, in0)
    @alias(i_start, in1)
    @alias(i_end, in2)
    
    __set_i_for_segment_start_end<<<({n_pts}+{threads}-1)/{threads}, {threads}>>>(
          ray_id_p, {n_pts}, i_start_p, i_end_p);
    i_end_p[ray_id_p[{n_pts}-1]] = {n_pts};
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error in __set_i_for_segment_start_end: %s\\n", cudaGetErrorString(err));
    
    ''')
    
    
    
    
    
    jt.code(inputs=[alpha,weight,T,alphainv_last,i_start,i_end],outputs=[alpha,weight,T,alphainv_last,i_start,i_end],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <iostream>
using  jittor::int64;

namespace{

template <typename scalar_t>
__global__ void alpha2weight_cuda_kernel(
    scalar_t* __restrict__ alpha,
    const int n_rays,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ T,
    scalar_t* __restrict__ alphainv_last,
    int64* __restrict__ i_start,
    int64* __restrict__ i_end) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int i_s = i_start[i_ray];
    const int i_e_max = i_end[i_ray];

    float T_cum = 1.;
    int i;
    for(i=i_s; i<i_e_max; ++i) {
      T[i] = T_cum;
      weight[i] = T_cum * alpha[i];
      T_cum *= (1. - alpha[i]);
      if(T_cum<1e-3) {
        i+=1;
        break;
      }
    }
    i_end[i_ray] = i;
    alphainv_last[i_ray] = T_cum;
  }
}  
    
    ''',
    cuda_src=f'''
    @alias(alpha, in0)
    @alias(weight, in1)
    @alias(T, in2)
    @alias(alphainv_last, in3)    
    @alias(i_start, in4)
    @alias(i_end, in5)
    
    
    const int blocks = ({n_rays} + {threads} - 1) / {threads};
    
    alpha2weight_cuda_kernel<float32><<<blocks, threads>>>(
        alpha_p,
        {n_rays},
        weight_p,
        T_p,
        alphainv_last_p,
        i_start_p,
        i_end_p);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in alpha2weight: %s\\n", cudaGetErrorString(err));
    ''')
    
    return weight,T,alphainv_last,i_start,i_end 
    
    
    
def alpha2weight_backward(
      alpha, weight, T, alphainv_last,
      i_start, i_end, n_rays,
      grad_weights, grad_last):
    
    return jt.code(alpha.shape,alpha.dtype,[alpha, weight, T, alphainv_last,
      i_start, i_end, grad_weights, grad_last],
    cuda_header='''

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace{

template <typename scalar_t>
__global__ void alpha2weight_backward_cuda_kernel(
    scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ T,
    scalar_t* __restrict__ alphainv_last,
    int64* __restrict__ i_start,
    int64* __restrict__ i_end,
    const int n_rays,
    scalar_t* __restrict__ grad_weights,
    scalar_t* __restrict__ grad_last,
    scalar_t* __restrict__ grad) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int i_s = i_start[i_ray];
    const int i_e = i_end[i_ray];

    float back_cum = grad_last[i_ray] * alphainv_last[i_ray];
    for(int i=i_e-1; i>=i_s; --i) {
      grad[i] = grad_weights[i] * T[i] - back_cum / (1-alpha[i] + 1e-10);
      back_cum += grad_weights[i] * weight[i];
    }
  }
}
    
    
    ''',
    cuda_src=f'''
    @alias(alpha, in0)
    @alias(weight, in1)
    @alias(T, in2)
    @alias(alphainv_last, in3)    
    @alias(i_start, in4)
    @alias(i_end, in5)
    @alias(grad_weights, in6)
    @alias(grad_last, in7)
    @alias(grad, out0)
    
    const int threads = 512;
    const int blocks = ({n_rays} + threads - 1) / threads;
    
    alpha2weight_backward_cuda_kernel<float32><<<blocks, threads>>>(
        alpha_p,
        weight_p,
        T_p,
        alphainv_last_p,
        i_start_p,
        i_end_p,
        {n_rays},
        grad_weights_p,
        grad_last_p,
        grad_p);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
            printf("Error in alpha2weight_backward: %s\\n", cudaGetErrorString(err));
    ''')