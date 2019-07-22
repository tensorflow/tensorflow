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

//
// fall-back version
//
template <typename T, typename U>
__global__ void CustomL2NormFunctor_kernel_stage1(int N, int k, const T* in, float* out, float averager, const U* _eps)
{
    float eps = (float)*_eps;
    for(int idx = threadIdx.x + 1024*blockIdx.x; idx<N; idx+=1024*1024)
    {
      float sum = 0;
      float sumsq = 0;
      for(int i=0; i<k; i++)
      {
        float x = (float)in[idx*k+i];
        sum += x;
        sumsq += x*x;
      }

      sumsq -= sum*sum*averager;
      float mean = sum*averager;
      float sigma = sumsq*averager;
      sigma = rsqrt(sigma+eps);

      out[idx*2+0] = mean;
      out[idx*2+1] = sigma;
  }
}

template <typename T, typename U>
__global__ void CustomL2NormFunctor_kernel_stage2(int N, int k, const T* in, const float* temp, T* out, const U* _bias, const U* _scale)
{
    for(int idx = threadIdx.x + 1024*blockIdx.x; idx<N; idx+=1024*1024)
    {
      float mean = temp[idx*2+0];
      float sigma = temp[idx*2+1];
      for(int i=0; i<k; i++)
        out[idx*k+i] = (T)(((float)in[idx*k+i] - mean) * (float)_scale[i] * sigma + (float)_bias[i]);
    }
}

template <typename T, typename U>
__global__ void CustomL2NormGradFunctor_kernel_stage2(int N, int k, const T* in, const T* out_grad, const float* temp, T* out, float a, 
  const U* _bias, const U* _scale)
{
  //float scale = (float)*_scale;
  for(int idx = threadIdx.x + 1024*blockIdx.x; idx<N; idx+=1024*1024)
  {
    float mean = temp[idx*2+0];
    float sigma = temp[idx*2+1];
    T* op = out+idx*k;
    const T* ip = in+idx*k;
    const T* ogp = out_grad+idx*k;
    float s1 = 0;
    float s2 = 0;
    for(int y=0; y<k; y++)
    {
      s1 += (float)ogp[y] * (float)_scale[y];
      s2 += (float)ogp[y] * (float)_scale[y] * ((float)ip[y]-mean);
    }

    s1 *= a*sigma;
    s2 *= a*sigma*sigma*sigma;

    for(int y=0; y<k; y++)
        op[y] = (T) ((float)ogp[y] * (float)_scale[y] * sigma - s1 - s2*((float)ip[y]-mean));
  }
}

//
// optimized version for k<=1024
//
template <typename T, typename U>
__global__ void CustomL2NormFunctor_kernel_stage1_v2(int N, int k, const T* in, float* out, float averager, const U* _eps)
{
    //int k = blockDim.x;
    int i = threadIdx.x;
    int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
    if(idx>=N)
      return;
    float eps = (float)*_eps;
    if(i==0)
    {
      out[idx*2+0] = 0;
      out[idx*2+1] = 0;
    }
    //__syncthreads();
    
    float sum = (i<k) ? (float)in[idx*k+i] : 0.0f;
    float sumsq = sum*sum;
    for(int m=1; m<32; m*=2)
    {
      sum += __shfl_xor_sync((unsigned)-1, sum, m);
      sumsq += __shfl_xor_sync((unsigned)-1, sumsq, m);
    }
    __syncthreads();
    if((i & 31)==0)
    {
      atomicAdd(out+idx*2+0, sum);
      atomicAdd(out+idx*2+1, sumsq);
    }
    __syncthreads();
    if(i==0)
    {
      sum = out[idx*2+0];
      sumsq = out[idx*2+1];
      sumsq -= sum*sum*averager;
      float mean = sum*averager;
      float sigma = sumsq*averager;
      sigma = rsqrt(sigma+eps);

      out[idx*2+0] = mean;
      out[idx*2+1] = sigma;
    }
 
}

template <typename T, typename U>
__global__ void CustomL2NormFunctor_kernel_stage2_v2(int N, const T* in, const float* temp, T* out, const U* _bias, const U* _scale)
{
  int k = blockDim.x;
  int i = threadIdx.x;
  int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
  if(idx>=N)
    return;
  float mean = temp[idx*2+0];
  float sigma = temp[idx*2+1];
  out[idx*k+i] = (T) (((float)in[idx*k+i] - mean) * (float)_scale[i] * sigma + (float)_bias[i]);
}

template <typename T, typename U>
__global__ void CustomL2NormGradFunctor_kernel_stage2_v2(int N, int k, const T* in, const T* out_grad, float* temp, T* out, float a,   const U* _bias, const U* _scale)
{
    int i = threadIdx.x;
    int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
    if(idx>=N)
      return;
    float mean = temp[idx*2+0];
    float sigma = temp[idx*2+1];
    T* op = out+idx*k;
    const T* ip = in+idx*k;
    const T* ogp = out_grad+idx*k;
    float s1 = 0;
    float s2 = 0;
    if(i<k)
    {
      s1 = (float)ogp[i] * (float)_scale[i];
      s2 = s1 * ((float)ip[i]-mean);
    }
    __syncthreads();
    if(i==0)
    {
      temp[idx*2+0]=0;
      temp[idx*2+1]=0;
    }
    for(int m=1; m<32; m*=2)
    {
      s1 += __shfl_xor_sync((unsigned)-1, s1, m);
      s2 += __shfl_xor_sync((unsigned)-1, s2, m);
    }
    __syncthreads();
    if((i & 31)==0)
    {
      atomicAdd(temp+idx*2+0, s1);
      atomicAdd(temp+idx*2+1, s2);
    }
    __syncthreads();
    s1 = temp[idx*2+0];
    s2 = temp[idx*2+1];
    s1 *= a*sigma;
    s2 *= a*sigma*sigma*sigma;
    if(i<k)
        op[i] = (T) ((float)ogp[i] * (float)_scale[i] * sigma - s1 - s2*((float)ip[i]-mean));
}

template <typename T, typename U>
void CustomL2NormFunctor<Eigen::GpuDevice, T, U>::operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, float* temp, T* out,
    const U* eps, const U* bias, const U* scale)
{
  uint64_t blocks = max(1ul,min(1024ul, (N+1023)>>10));

  if(k>=2 && k<=1024)
  {
    dim3 threads, blocks;
    int thrRound = (k+31)&~31;
    threads=dim3(thrRound, 1024/thrRound, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    CustomL2NormFunctor_kernel_stage1_v2<T,U> <<<blocks, threads, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);

    threads = dim3(k, 1024/k, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    if(blocks.x>1024)
      blocks=dim3(1024, (blocks.x+1023)/1024, 1);
    CustomL2NormFunctor_kernel_stage2_v2<T,U> <<<blocks, threads, 0, d.stream()>>> (N, in, temp, out, bias, scale);
  }
  else
  {
    uint64_t blocks = max(1ul,min(1024ul, (N+1023)>>10));
    CustomL2NormFunctor_kernel_stage1<T,U> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);
    CustomL2NormFunctor_kernel_stage2<T,U> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, temp, out, bias, scale);
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
  if(k>=2 && k<=1024)
  {
    dim3 threads, blocks;
    int thrRound = (k+31)&~31;
    threads=dim3(thrRound, 1024/thrRound, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    if(blocks.x>1024)
      blocks=dim3(1024, (blocks.x+1023)/1024, 1);
    CustomL2NormFunctor_kernel_stage1_v2<T,U> <<<blocks, threads, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);
    CustomL2NormGradFunctor_kernel_stage2_v2<T,U> <<<blocks, threads, 0, d.stream()>>> (N, k, in, outgrad, temp, out, 1./k, bias, scale);
  }
  else
  {
    uint64_t blocks = max(1ul,min(1024ul, (N+1023)>>10));
    CustomL2NormFunctor_kernel_stage1<T,U> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);
    CustomL2NormGradFunctor_kernel_stage2<T,U> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, outgrad, temp, out, 1./k, bias, scale);
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

/* Special case for d <= 1024 (no need for a loop over d2) */
template <typename T>
__global__ void CustomDropoutFunctor3_v2_kernel(const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2,
    int s0, int s1, int s2,
    int r0, int r1, int r2)
{
    T threshold = (T)*pthr;
    T scale = T(1.)/(T(1.)-threshold);
    int i=blockIdx.x+blockIdx.y*gridDim.x;
    const T* ip = in + i*s0 + threadIdx.y*s1 + threadIdx.x;
    const T* rp = rng + i*r0 + threadIdx.y*r1 + threadIdx.x*r2;
    T* op = out + i*s0 + threadIdx.y*s1 + threadIdx.x;
    s0 *= gridDim.x*gridDim.y;
    r0 *= gridDim.x*gridDim.y;
    s1 *= blockDim.y;
    r1 *= blockDim.y;

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
  //  printf("fallback: %d x %d x %d\n", d0, d1, d2);
    CustomDropoutFunctor2<Eigen::GpuDevice,T>()(d, in, rng, out, pthr, d1, d2, s1, s2, r1, r2);
  }
  else
  /* if(d2<=1024)
  {
    dim3 threads(d2,1024/d2,1);
    dim3 blocks(min(1024,d0),min(1024,(d0+1023)/1024),1);
  //   printf("v2: %d x %d x %d -> %d %d\n", d0, d1, d2, threads.x, threads.y);
    CustomDropoutFunctor3_v2_kernel<<<blocks,threads,0, d.stream()>>> (in, rng, out, pthr, d0, d1, d2, s0, s1, s2, r0, r1, r2);
  }
  else
    */
  {
    int threads_x = min(d2, 1024);
    int threads_y = min(d1, 1024/threads_x);
    dim3 threads(threads_x, threads_y, 1);
    dim3 blocks((d2+1023)/1024, min(65536,d0),min(65536,(d0+65535)/65536));
  //  printf("v1: %d x %d x %d -> %d %d\n", d0, d1, d2, threads_x, threads_y);
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
  //  printf("fallback: %d x %d x %d\n", d0, d1, d2);
    CustomDropoutFunctor3<Eigen::GpuDevice,T>()(d, in, rng, out, pthr, d1, d2, d3, s1, s2, s3, r1, r2, r3);
  }
  else if(d3 == 1)
  {
    CustomDropoutFunctor3<Eigen::GpuDevice,T>()(d, in, rng, out, pthr, d0, d1, d2, s0, s1, s2, r0, r1, r2);
  }
  else
  /* if(d2<=1024)
  {
    dim3 threads(d2,1024/d2,1);
    dim3 blocks(min(1024,d0),min(1024,(d0+1023)/1024),1);
  //   printf("v2: %d x %d x %d -> %d %d\n", d0, d1, d2, threads.x, threads.y);
    CustomDropoutFunctor3_v2_kernel<<<blocks,threads,0, d.stream()>>> (in, rng, out, pthr, d0, d1, d2, s0, s1, s2, r0, r1, r2);
  }
  else
    */
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

