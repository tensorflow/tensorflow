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

//template struct CustomL2NormFunctor<Eigen::GpuDevice, float>;

template <typename T>
__global__ void CustomL2NormFunctor_kernel_stage1(int N, int k, const T* in, float* out, float averager, const float* _eps)
{
    float eps = *_eps;
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

template <typename T>
__global__ void CustomL2NormFunctor_kernel_stage1_v2(int N, int k, const T* in, float* out, float averager, const float* _eps)
{
    //int k = blockDim.x;
    int i = threadIdx.x;
    int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
    if(idx>=N)
      return;
    float eps = *_eps;
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


template <typename T>
__global__ void CustomL2NormFunctor_kernel_stage2(int N, int k, const T* in, const float* temp, T* out, const float* _bias, const float* _scale)
{
    for(int idx = threadIdx.x + 1024*blockIdx.x; idx<N; idx+=1024*1024)
    {
      float mean = temp[idx*2+0];
      float sigma = temp[idx*2+1];
      for(int i=0; i<k; i++)
        out[idx*k+i] = (T)(((float)in[idx*k+i] - mean) * _scale[i] * sigma + _bias[i]);
    }
}

template <typename T>
__global__ void CustomL2NormFunctor_kernel_stage2_v2(int N, const T* in, const float* temp, T* out, const float* _bias, const float* _scale)
{
  int k = blockDim.x;
  int i = threadIdx.x;
  int idx = threadIdx.y + blockDim.y*(blockIdx.x + 1024*blockIdx.y);
  if(idx>=N)
    return;
  float mean = temp[idx*2+0];
  float sigma = temp[idx*2+1];
  out[idx*k+i] = (T) (((float)in[idx*k+i] - mean) * _scale[i] * sigma + _bias[i]);
}

template <typename T>
__global__ void CustomL2NormGradFunctor_kernel_stage2_v2(int N, int k, const T* in, const T* out_grad, float* temp, T* out, float a, 
  const float* _bias, const float* _scale)
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
      s1 = (float)ogp[i] * _scale[i];
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
        op[i] = (T) ((float)ogp[i] * _scale[i] * sigma - s1 - s2*((float)ip[i]-mean));
}

template <typename T>
__global__ void CustomL2NormGradFunctor_kernel_stage2(int N, int k, const T* in, const T* out_grad, const float* temp, T* out, float a, 
  const float* _bias, const float* _scale)
{
  float scale = *_scale;
  //float inv_scale = a / (scale*scale);
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
      s1 += (float)ogp[y] * _scale[y];
      s2 += (float)ogp[y] * _scale[y] * ((float)ip[y]-mean);
    }

    s1 *= a*sigma;
    s2 *= a*sigma*sigma*sigma;

    for(int y=0; y<k; y++)
        op[y] = (T) ((float)ogp[y] * _scale[y] * sigma - s1 - s2*((float)ip[y]-mean));
  }
}

template <typename T>
void CustomL2NormFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, float* temp, T* out,
    const float* eps, const float* bias, const float* scale)
{
  uint64_t blocks = max(1ul,min(1024ul, (N+1023)>>10));
  //printf("CustomL2NormFunctor %lu x %lu\n", N, k);

  if(k>=2 && k<=1024)
  {
    dim3 threads, blocks;
    int thrRound = (k+31)&~31;
    threads=dim3(thrRound, 1024/thrRound, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    CustomL2NormFunctor_kernel_stage1_v2<T> <<<blocks, threads, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);

    threads = dim3(k, 1024/k, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    if(blocks.x > 1024*1024)
    {
      printf("Too many threads\n");
      exit(-1);
    }
    if(blocks.x>1024)
      blocks=dim3(1024, (blocks.x+1023)/1024, 1);
    CustomL2NormFunctor_kernel_stage2_v2<T> <<<blocks, threads, 0, d.stream()>>> (N, in, temp, out, bias, scale);
  }
  else
  {
    uint64_t blocks = max(1ul,min(1024ul, (N+1023)>>10));
    CustomL2NormFunctor_kernel_stage1<T> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);
    CustomL2NormFunctor_kernel_stage2<T> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, temp, out, bias, scale);
  }
}


template <typename T>
void CustomL2NormGradFunctor<Eigen::GpuDevice, T>::operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, 
    const T* outgrad,
    float* temp,
    //T* temp2,
    T* out,
    const float* eps, const float* bias, const float* scale)
{
  if(k>=2 && k<=1024)
  {
    dim3 threads, blocks;
    int thrRound = (k+31)&~31;
    threads=dim3(thrRound, 1024/thrRound, 1);
    blocks=dim3( max(1ul, uint64_t((N+threads.y-1) / threads.y)), 1, 1);
    CustomL2NormFunctor_kernel_stage1_v2<T> <<<blocks, threads, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);
    CustomL2NormGradFunctor_kernel_stage2_v2<T> <<<blocks, threads, 0, d.stream()>>> (N, k, in, outgrad, temp, out, 1./k, bias, scale);
  }
  else
  {
    uint64_t blocks = max(1ul,min(1024ul, (N+1023)>>10));
    CustomL2NormFunctor_kernel_stage1<T> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, temp, 1./k, eps);
    CustomL2NormGradFunctor_kernel_stage2<T> <<<blocks, 1024, 0, d.stream()>>> (N, k, in, outgrad, temp, out, 1./k, bias, scale);
  }
}


template struct CustomL2NormFunctor<Eigen::GpuDevice, float>;
template struct CustomL2NormGradFunctor<Eigen::GpuDevice, float>;
template struct CustomL2NormFunctor<Eigen::GpuDevice, Eigen::half>;
template struct CustomL2NormGradFunctor<Eigen::GpuDevice, Eigen::half>;

}

#endif

