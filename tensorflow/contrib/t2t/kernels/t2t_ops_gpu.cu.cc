#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "t2t_ops.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <inttypes.h>

#include <cuda_fp16.h>

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

__device__ float2& operator+=(float2& a, const float2& b)
{
  a.x+=b.x;
  a.y+=b.y;
  return a;
}

__device__ void add_shfl_xor(float2& var, unsigned m)
{
  var.x += __shfl_xor_sync((unsigned)-1, var.x, m);
  var.y += __shfl_xor_sync((unsigned)-1, var.y, m);
}

template <class T>
__device__ void reduce_sum_across_block(T& v)
{
    assert(!(blockDim.x & 31));
    int tid = threadIdx.x & 31;
    int xid = threadIdx.x >> 5;
    int xdim = blockDim.x >> 5;
    int yid = threadIdx.y;
    __shared__ T sums[32][32]; // 8 kb for 2x float - overkill but simplifies coding
    T* p = &sums[yid][xid];

    for(int m=1; m<32; m*=2)
      add_shfl_xor(v, m);

    if(tid==0)
      p[0] = v;
   __syncthreads();
    for(int m=1; m<32; m*=2)
    {
      if((tid==0) && (!(xid & m)) && (xid+m<xdim))
          p[0] += p[m];
      __syncthreads();
    }
    v = sums[yid][0];
}

template <typename T, typename U>
__global__ void CustomL2NormFunctor_kernel_stage1(int N, int k, const T* in, float* out, float averager, const U* _eps)
{
    int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
    if(idx>=N)
      return;
    float eps = (float)*_eps;
    if(threadIdx.x==0)
    {
      out[idx*2+0] = 0;
      out[idx*2+1] = 0;
    }
    float2 sum = {0,0};
    for(int i=threadIdx.x; i<k; i+=blockDim.x)
    { 
      float t = (float)in[idx*k+i];
      sum.x += t;
      sum.y += t*t;
    }
    reduce_sum_across_block(sum);
    if(threadIdx.x==0)
    {
      sum.y -= sum.x*sum.x*averager;
      float mean = sum.x*averager;
      float sigma = sum.y*averager;
      sigma = rsqrt(sigma+eps);
      out[idx*2+0] = mean;
      out[idx*2+1] = sigma;
    }
}

template <typename T, typename U>
__global__ void CustomL2NormFunctor_kernel_stage2(int N, int k, const T* in, const float* temp, T* out, const U* _bias, const U* _scale)
{
  int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
  if(idx>=N)
    return;
  float mean = temp[idx*2+0];
  float sigma = temp[idx*2+1];
  for(int i = threadIdx.x; i<k; i+=blockDim.x)
    out[idx*k+i] = (T) (((float)in[idx*k+i] - mean) * (float)_scale[i] * sigma + (float)_bias[i]);
}

template <typename T, typename U>
__global__ void CustomL2NormGradFunctor_kernel_stage2(int N, int k, const T* in, const T* out_grad, float* temp, T* out, float a,   const U* _bias, const U* _scale)
{
    int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
    if(idx>=N)
      return;
    float mean = temp[idx*2+0];
    float sigma = temp[idx*2+1];
    T* op = out+idx*k;
    const T* ip = in+idx*k;
    const T* ogp = out_grad+idx*k;
    float2 s = {0,0};
    for(int i=threadIdx.x; i<k; i+=blockDim.x)
    {
      float t = (float)ogp[i] * (float)_scale[i];
      s.x += t;
      s.y += t * ((float)ip[i]-mean);
    }
    reduce_sum_across_block(s);
    float s1 = s.x;
    float s2 = s.y;
    s1 *= a*sigma;
    s2 *= a*sigma*sigma*sigma;
    for(int i=threadIdx.x; i<k; i+=blockDim.x)
        op[i] = (T) ((float)ogp[i] * (float)_scale[i] * sigma - s1 - s2*((float)ip[i]-mean));
}

template <typename T, typename U>
void CustomL2NormFunctor<Eigen::GpuDevice, T, U>::operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, float* temp, T* out,
    const U* eps, const U* bias, const U* scale)
{
  uint64_t blocks = max(1ul,min(1024ul, (N+1023)>>10));

  {
    dim3 threads, blocks;
    int thrRound = min(1024ul, (k+31)&~31);
    threads=dim3(thrRound, 1024/thrRound, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    CustomL2NormFunctor_kernel_stage1<T,U> <<<blocks, threads, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);

    thrRound = min(1024ul, k);
    threads = dim3(thrRound, 1024/thrRound, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    if(blocks.x>1024)
      blocks=dim3(1024, (blocks.x+1023)/1024, 1);
    CustomL2NormFunctor_kernel_stage2<T,U> <<<blocks, threads, 0, d.stream()>>> (N, k, in, temp, out, bias, scale);
  }
}


template <typename T, typename U>
void CustomL2NormGradFunctor<Eigen::GpuDevice, T, U>::operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, 
    const T* outgrad,
    float* temp,
    T* out,
    const U* eps, const U* bias, const U* scale)
{
  {
    dim3 threads, blocks;
    int thrRound = min(1024ul, (k+31)&~31);
    threads=dim3(thrRound, 1024/thrRound, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    if(blocks.x>1024)
      blocks=dim3(1024, (blocks.x+1023)/1024, 1);
    CustomL2NormFunctor_kernel_stage1<T,U> <<<blocks, threads, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);
    CustomL2NormGradFunctor_kernel_stage2<T,U> <<<blocks, threads, 0, d.stream()>>> (N, k, in, outgrad, temp, out, 1./k, bias, scale);
  }
}

template <typename T>
__global__ void CustomDropoutFunctor1_kernel(const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d)
{
    T threshold = (T)*pthr;
    T scale = T(1.)/(T(1.)-threshold);
    int off = threadIdx.x + blockIdx.x * 1024;
    if(off < d)
        out[off] = in[off] * (rng[off]>=threshold ? scale : (T)0.0);
}


template <typename T>
__global__ void CustomDropoutFunctor2_kernel(const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1,
    int s0, int s1,
    int r0, int r1)
{
    T threshold = (T)*pthr;
    T scale = T(1.)/(T(1.)-threshold);

    for(int i=blockIdx.x+blockIdx.y*gridDim.x; i<d0; i+=gridDim.x*gridDim.y)
      for(int j=threadIdx.x; j<d1; j+=1024)
          out[i*s0+j] = in[i*s0+j] * (rng[i*r0+j*r1]>=threshold ? scale : (T)0.0);
}

// TODO: all the explicit loop unrolling may be unnecessary
template <typename T>
__global__ void CustomDropoutFunctor3_kernel(const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2,
    int s0, int s1, int s2,
    int r0, int r1, int r2)
{
    T threshold = (T)*pthr;
    T scale = T(1.)/(T(1.)-threshold);
    int i=blockIdx.y+blockIdx.z*gridDim.y;
    int k = threadIdx.x + blockIdx.x * 1024;
    if(k>=d2)
      return;
    const T* ip = in + i*s0 + threadIdx.y*s1 + k;
    const T* rp = rng + i*r0 + threadIdx.y*r1 + k*r2;
    T* op = out + i*s0 + threadIdx.y*s1 + k;
/*
    for(; i<d0; i+=gridDim.x*gridDim.y)
      for(int j=threadIdx.y; j<d1; j+=blockDim.y)
        for(int k=threadIdx.x; k<d2; k+=blockDim.x)
          out[i*s0 + j*s1 + k] = in[i*s0 + j*s1 + k] * (rng[i*r0 + j*r1 + k*r2]>=threshold ? scale : (T)0.0); 
*/
    
    s0 *= gridDim.x*gridDim.y;
    r0 *= gridDim.x*gridDim.y;
    s1 *= blockDim.y;
    r1 *= blockDim.y;
    r2 *= blockDim.x;
    for(; i<d0; i+=gridDim.x*gridDim.y)
    {
      const T* ipp = ip;
      const T* rpp = rp;
      T* opp = op;
      for(int j=threadIdx.y; j<d1; j+=blockDim.y)
      {
        opp[0] = ipp[0] * (rpp[0]>=threshold ? scale : (T)0.0);

        ipp += s1;
        opp += s1;
        rpp += r1;
      }
      ip += s0;
      op += s0;
      rp += r0;
    }    
}


// TODO: all the explicit loop unrolling may be unnecessary
template <typename T>
__global__ void CustomDropoutFunctor4_kernel(const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2, int d3,
    int s0, int s1, int s2, int s3,
    int r0, int r1, int r2, int r3)
{
    T threshold = (T)*pthr;
    T scale = T(1.)/(T(1.)-threshold);
    int c1 = blockIdx.y;
    int c0 = blockIdx.z;
    int c3 = threadIdx.x + blockIdx.x * 1024;
    int c2 = threadIdx.y;
    if(c3>=d3)
      return;
    in += c3;
    out += c3;
    rng += c3*r3;
    for(; c0 < d0; c0+=gridDim.z)
      for(; c1 < d1; c1+=gridDim.y)
        for(; c2 < d2; c2+=blockDim.y)
        {
          int off1 = c0*s0 + c1*s1 + c2*s2;
          int off2 = c0*r0 + c1*r1 + c2*r2;
          out[off1] = in[off1] * (rng[off2]>=threshold ? scale : (T)0.0);
        }
}

template <typename T>
void CustomDropoutFunctor2<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1,
    int s0, int s1,
    int r0, int r1
    )
{
    dim3 threads(min(d1,1024),1,1);
    dim3 blocks(min(1024,d0),min(1024,(d0+1023)/1024),1);
    CustomDropoutFunctor2_kernel<<<blocks,threads,0, d.stream()>>> (in, rng, out, pthr, d0, d1, s0, s1, r0, r1);
}

template <typename T>
void CustomDropoutFunctor3<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2,
    int s0, int s1, int s2,
    int r0, int r1, int r2
    )
{
  if(r0==s0 && r1==s1 && r2==s2)
  {
    int dim = d0*d1*d2;
    CustomDropoutFunctor1_kernel<<<(dim+1023)/1024,min(dim,1024),0, d.stream()>>> (in, rng, out, pthr, dim);
  }
  else if(d0 == 1)
  {
    CustomDropoutFunctor2<Eigen::GpuDevice,T>()(d, in, rng, out, pthr, d1, d2, s1, s2, r1, r2);
  }
  else
  {
    int threads_x = min(d2, 1024);
    int threads_y = min(d1, 1024/threads_x);
    dim3 threads(threads_x, threads_y, 1);
    dim3 blocks((d2+1023)/1024, min(65536,d0),min(65536,(d0+65535)/65536));
    CustomDropoutFunctor3_kernel<<<blocks,threads,0, d.stream()>>> (in, rng, out, pthr, d0, d1, d2, s0, s1, s2, r0, r1, r2);
  }
}


template <typename T>
void CustomDropoutFunctor4<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2, int d3,
    int s0, int s1, int s2, int s3,
    int r0, int r1, int r2, int r3
    )
{
  if(r0==s0 && r1==s1 && r2==s2 && r3==s3)
  {
    int dim = d0*d1*d2*d3;
    CustomDropoutFunctor1_kernel<<<(dim+1023)/1024,min(dim,1024),0, d.stream()>>> (in, rng, out, pthr, dim);
  }
  else if(d0 == 1)
  {
    CustomDropoutFunctor3<Eigen::GpuDevice,T>()(d, in, rng, out, pthr, d1, d2, d3, s1, s2, s3, r1, r2, r3);
  }
  else if(d3 == 1)
  {
    CustomDropoutFunctor3<Eigen::GpuDevice,T>()(d, in, rng, out, pthr, d0, d1, d2, s0, s1, s2, r0, r1, r2);
  }
  else
  {
    int threads_x = min(d3, 1024);
    int threads_y = min(d2, 1024/threads_x);
    dim3 threads(threads_x, threads_y, 1);
    dim3 blocks((d3+1023)/1024, min(65536,d1),min(65536,d0));
    CustomDropoutFunctor4_kernel<<<blocks,threads,0, d.stream()>>> (in, rng, out, pthr, d0, d1, d2, d3, s0, s1, s2, s3, r0, r1, r2, r3);
  }
}


template struct CustomL2NormFunctor<Eigen::GpuDevice, float, float>;
template struct CustomL2NormGradFunctor<Eigen::GpuDevice, float, float>;
template struct CustomL2NormFunctor<Eigen::GpuDevice, Eigen::half, float>;
template struct CustomL2NormGradFunctor<Eigen::GpuDevice, Eigen::half, float>;
template struct CustomL2NormFunctor<Eigen::GpuDevice, Eigen::half, Eigen::half>;
template struct CustomL2NormGradFunctor<Eigen::GpuDevice, Eigen::half, Eigen::half>;

template struct CustomDropoutFunctor2<Eigen::GpuDevice, float>;
template struct CustomDropoutFunctor2<Eigen::GpuDevice, Eigen::half>;
template struct CustomDropoutFunctor3<Eigen::GpuDevice, float>;
template struct CustomDropoutFunctor3<Eigen::GpuDevice, Eigen::half>;
template struct CustomDropoutFunctor4<Eigen::GpuDevice, float>;
template struct CustomDropoutFunctor4<Eigen::GpuDevice, Eigen::half>;

}

#endif

