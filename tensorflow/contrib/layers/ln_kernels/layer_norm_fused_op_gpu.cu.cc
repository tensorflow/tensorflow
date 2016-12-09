/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <stdio.h>
#include <math.h>
#include <algorithm>

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "./layer_norm_fused_op.h"

#if !defined(_MSC_VER)
#define UNROLL _Pragma("unroll")
#else
#define UNROLL 
#endif

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif


#define MAX_GRID_SIZE 480
inline int get_num_blocks(const int n_slices,const int slice_per_block)
{
  const int _num_blocks = n_slices/slice_per_block;
  if(_num_blocks*slice_per_block==n_slices)
    return _num_blocks;
  else
    return _num_blocks+1;
}

inline int get_block_size(const int slice_size,int& block_size,
  int& mult)
{
  const int _warp_size=32;
  int _block_size = _warp_size;
  int _mult = slice_size/_block_size;
  mult = _mult*_block_size>=slice_size?_mult:_mult+1;
  while (mult>5)
  {
    _block_size+=_warp_size;
    _mult = slice_size/_block_size;
    mult = _mult*_block_size>=slice_size?_mult:_mult+1;
  }
  block_size=_block_size;
  return 0;
}

template <typename T>
__global__ void fillZeros(T* __restrict__ output,const int n_inputs)
{
  for(int thread_id=threadIdx.x+blockDim.x*blockIdx.x;thread_id<n_inputs;thread_id+=blockDim.x*gridDim.x)
    output[thread_id] = static_cast<T>(0.0f);
}

template <typename T>
__device__ __inline__ void warpSum(T& val1, T& val2)
{
  val1 += __shfl_xor(val1, 16);
  val2 += __shfl_xor(val2, 16);
  val1 += __shfl_xor(val1, 8);
  val2 += __shfl_xor(val2, 8);
  val1 += __shfl_xor(val1, 4);
  val2 += __shfl_xor(val2, 4);
  val1 += __shfl_xor(val1, 2);
  val2 += __shfl_xor(val2, 2);
  val1 += __shfl_xor(val1, 1);
  val2 += __shfl_xor(val2, 1);
}


template <typename T>
__device__ __inline__ T get_value(const T* index, const int bound_check,const int up_bound)
{
    if (bound_check < up_bound) return __ldg(index);
    else return static_cast<T>(0.0f);
}

namespace tensorflow {

  namespace 
  {

    typedef Eigen::GpuDevice GPUDevice;

    template<typename T, int mult>
    __global__ void LayerNormGPUKernel(const LayerNormFusedArgs args,
      const T* __restrict__ input,
        T* __restrict__ output,
        const int num_blocks
        )
    {
        const int in_depth = args.depth;
        const int slice_size = args.slice_size;
        const int n_inputs = args.n_inputs;
        const T epsilon = args.epsilon;

        const int tWarpIdx = threadIdx.x%warpSize;

        const int tSliceIdx = threadIdx.x%slice_size;

        extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
        T* mean_cache = (T*)my_smem;
        T* std_cache = (T*)&my_smem[sizeof(T)];

        const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);
        T inp[mult];
        int thread_id[mult];

        T sum;
        T sqSum;
        T mu;
        T rstd;

        for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
        {   
            sum = static_cast<T>(0.0f);
            sqSum = static_cast<T>(0.0f);

            if(tSliceIdx==0){
              mean_cache[0]=static_cast<T>(0.0f);
              std_cache[0]=static_cast<T>(0.0f);
            }
            __syncthreads();

            UNROLL for(int m=0;m<mult;m++)
            {
                thread_id[m] = bId*in_depth+m*blockDim.x+tSliceIdx;
            }

            UNROLL for(int m=0;m<mult;m++)
            {
                inp[m] = get_value<T>(input+thread_id[m],tSliceIdx+m*blockDim.x,in_depth);
            }

            UNROLL for(int m=0;m<mult;m++)
            {
                sum += inp[m]*i_n;
            }   
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
            {
                sum +=__shfl_xor(sum, mask);
            }
            if(tWarpIdx==0)
            {
                atomicAdd(&mean_cache[0],sum);
            }
            __syncthreads();

            mu = mean_cache[0];
            UNROLL for(int m=0;m<mult;m++)
            {
                if (tSliceIdx+m*blockDim.x<in_depth) sqSum += (inp[m]-mu)*(inp[m]-mu);
            }   
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
            {
                sqSum += __shfl_xor(sqSum, mask);
            }
            if(tWarpIdx==0)
            {
                atomicAdd(&std_cache[0],sqSum);
            }
            __syncthreads();
            if(tSliceIdx==0)
            {
                std_cache[0] = rsqrt(std_cache[0]*i_n+epsilon);
            }
            __syncthreads();
            rstd = std_cache[0];

            UNROLL for(int m=0;m<mult;m++)
            {
                if(tSliceIdx+m*blockDim.x<in_depth&&thread_id[m]<n_inputs)
                {
                  output[thread_id[m]] = (inp[m]-mu)*rstd;
                }
            }
            __syncthreads();
        }
    }

    //fused small LN kernel
    template<typename T>
    __global__ void LayerNormSmallGPUKernel(const LayerNormFusedArgs args,
      const T* __restrict__ input,
        T* __restrict__ output,
        const int num_blocks,const int slice_per_block
        )
    {
        const int slice_size = args.slice_size;
        const int in_depth = args.depth;
        const int n_inputs = args.n_inputs;
        const T epsilon = args.epsilon;

        const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

        const int slice_id = threadIdx.x/slice_size;
        const int tSliceIdx = threadIdx.x%slice_size;

        T mu;
        T rstd;
        
        for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
        {   
          mu = static_cast<T>(0.0f);
          rstd = static_cast<T>(0.0f);

          const int thread_id = (bId*slice_per_block+slice_id)*in_depth+tSliceIdx;
          // const T inp = 0;
          const T inp = get_value<T>(input+thread_id,tSliceIdx,in_depth);

          mu += inp*i_n;

          for (int mask = slice_size/2; mask > 0; mask /= 2) 
          {
              mu +=__shfl_xor(mu, mask);
          }

          if (tSliceIdx<in_depth) rstd += (inp-mu)*(inp-mu);

          for (int mask = slice_size/2; mask > 0; mask /= 2) 
          {
              rstd += __shfl_xor(rstd, mask);
          }

          rstd = rsqrt(rstd*i_n+epsilon);

          if (tSliceIdx<in_depth&& thread_id<n_inputs) output[thread_id] = (inp-mu)*rstd;
        }
    }

    template<typename T, int mult>
    __global__ void LayerNormBiasAddGPUKernel(const LayerNormFusedArgs args,
      const T* __restrict__ input,
      const T* __restrict__ beta,
        T* __restrict__ output,
        const int num_blocks
        )
    {
        const int in_depth = args.depth;
        const int slice_size = args.slice_size;
        const int n_inputs = args.n_inputs;
        const T epsilon = args.epsilon;

        const int tWarpIdx = threadIdx.x%warpSize;

        const int tSliceIdx = threadIdx.x%slice_size;

        extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
        T* mean_cache = (T*)my_smem;
        T* std_cache = (T*)&my_smem[sizeof(T)];

        const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);
        T inp[mult];
        T _beta[mult];
        int thread_id[mult];

        T sum;
        T sqSum;
        T mu;
        T rstd;

        UNROLL for(int m=0;m<mult;m++)
        {
          _beta[m] = get_value<T>(beta+tSliceIdx+m*blockDim.x,tSliceIdx+m*blockDim.x,in_depth);
        }

        for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
        {   
            sum = static_cast<T>(0.0f);
            sqSum = static_cast<T>(0.0f);

            if(tSliceIdx==0){
              mean_cache[0]=static_cast<T>(0.0f);
              std_cache[0]=static_cast<T>(0.0f);
            }
            __syncthreads();

            UNROLL for(int m=0;m<mult;m++)
            {
                thread_id[m] = bId*in_depth+m*blockDim.x+tSliceIdx;
            }

            UNROLL for(int m=0;m<mult;m++)
            {
                inp[m] = get_value<T>(input+thread_id[m],tSliceIdx+m*blockDim.x,in_depth);
            }

            UNROLL for(int m=0;m<mult;m++)
            {
                sum += inp[m]*i_n;
            }   
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
            {
                sum +=__shfl_xor(sum, mask);
            }
            if(tWarpIdx==0)
            {
                atomicAdd(&mean_cache[0],sum);
            }
            __syncthreads();

            mu = mean_cache[0];
            UNROLL for(int m=0;m<mult;m++)
            {
                if (tSliceIdx+m*blockDim.x<in_depth) sqSum += (inp[m]-mu)*(inp[m]-mu);
            }   
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
            {
                sqSum += __shfl_xor(sqSum, mask);
            }
            if(tWarpIdx==0)
            {
                atomicAdd(&std_cache[0],sqSum);
            }
            __syncthreads();
            if(tSliceIdx==0)
            {
                std_cache[0] = rsqrt(std_cache[0]*i_n+epsilon);
            }
            __syncthreads();
            rstd = std_cache[0];

            UNROLL for(int m=0;m<mult;m++)
            {
                if(tSliceIdx+m*blockDim.x<in_depth&&thread_id[m]<n_inputs)
                {
                  output[thread_id[m]] = (inp[m]-mu)*rstd+_beta[m];
                }
            }
            __syncthreads();
        }
    }

    //fused small LN kernel
    template<typename T>
    __global__ void LayerNormBiasAddSmallGPUKernel(const LayerNormFusedArgs args,
      const T* __restrict__ input,
      const T* __restrict__ beta,
        T* __restrict__ output,
        const int num_blocks,const int slice_per_block
        )
    {
        const int slice_size = args.slice_size;
        const int in_depth = args.depth;
        const int n_inputs = args.n_inputs;
        const T epsilon = args.epsilon;

        const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

        const int slice_id = threadIdx.x/slice_size;
        const int tSliceIdx = threadIdx.x%slice_size;

        const T _beta = get_value<T>(beta+tSliceIdx,tSliceIdx,in_depth);

        T mu;
        T rstd;
        
        for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
        {   
          mu = static_cast<T>(0.0f);
          rstd = static_cast<T>(0.0f);

          const int thread_id = (bId*slice_per_block+slice_id)*in_depth+tSliceIdx;
          // const T inp = 0;
          const T inp = get_value<T>(input+thread_id,tSliceIdx,in_depth);

          mu += inp*i_n;

          for (int mask = slice_size/2; mask > 0; mask /= 2) 
          {
              mu +=__shfl_xor(mu, mask);
          }

          if (tSliceIdx<in_depth) rstd += (inp-mu)*(inp-mu);

          for (int mask = slice_size/2; mask > 0; mask /= 2) 
          {
              rstd += __shfl_xor(rstd, mask);
          }

          rstd = rsqrt(rstd*i_n+epsilon);

          if (tSliceIdx<in_depth&& thread_id<n_inputs) output[thread_id] = (inp-mu)*rstd+_beta;
        }
    }

    template<typename T, int mult>
    __global__ void LayerNormFusedGPUKernel(const LayerNormFusedArgs args,
      const T* __restrict__ input,
      const T* __restrict__ gamma,
      const T* __restrict__ beta,
        T* __restrict__ output,
        const int num_blocks
        )
    {
        const int in_depth = args.depth;
        const int slice_size = args.slice_size;
        const int n_inputs = args.n_inputs;
        const T epsilon = args.epsilon;

        const int tWarpIdx = threadIdx.x%warpSize;

        const int tSliceIdx = threadIdx.x%slice_size;

        extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
        T* mean_cache = (T*)my_smem;
        T* std_cache = (T*)&my_smem[sizeof(T)];

        const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);
        T inp[mult];
        T _gamma[mult];
        T _beta[mult];
        int thread_id[mult];

        T sum;
        T sqSum;
        T mu;
        T rstd;

        UNROLL for(int m=0;m<mult;m++)
        {
          _gamma[m] = get_value<T>(gamma+tSliceIdx+m*blockDim.x,tSliceIdx+m*blockDim.x,in_depth);
          _beta[m] = get_value<T>(beta+tSliceIdx+m*blockDim.x,tSliceIdx+m*blockDim.x,in_depth);
        }

        for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
        {   
            sum = static_cast<T>(0.0f);
            sqSum = static_cast<T>(0.0f);

            if(tSliceIdx==0){
              mean_cache[0]=static_cast<T>(0.0f);
              std_cache[0]=static_cast<T>(0.0f);
            }
            __syncthreads();

            UNROLL for(int m=0;m<mult;m++)
            {
                thread_id[m] = bId*in_depth+m*blockDim.x+tSliceIdx;
            }

            UNROLL for(int m=0;m<mult;m++)
            {
                // inp[m] = __ldg(input+thread_id[m]);
                inp[m] = get_value<T>(input+thread_id[m],tSliceIdx+m*blockDim.x,in_depth);
                // if(blockIdx.x==0&&threadIdx.x==0) printf("m:%d,bId:%d,tId:%d,inp:%f\n",m,bId,thread_id[m],inp[m]);
            }

            UNROLL for(int m=0;m<mult;m++)
            {
                sum += inp[m]*i_n;
            }   
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
            {
                sum +=__shfl_xor(sum, mask);
            }
            // if(blockIdx.x==0&&tWarpIdx==0)printf("wi:%d,sum:%f,psmu:%f\n",threadIdx.x/warpSize, sum,mean_cache[slice_id]);
            if(tWarpIdx==0)
            {
                atomicAdd(&mean_cache[0],sum);
            }
            // if(blockIdx.x==0&&tWarpIdx==0)printf("--wi:%d,sum:%f,psmu:%f\n",threadIdx.x/warpSize, sum,mean_cache[slice_id]);
            __syncthreads();

            mu = mean_cache[0];
            // if(threadIdx.x==0&&blockIdx.x==0)printf("mu:%f\n", mu);
            UNROLL for(int m=0;m<mult;m++)
            {
                if (tSliceIdx+m*blockDim.x<in_depth) sqSum += (inp[m]-mu)*(inp[m]-mu);
            }   
            for (int mask = warpSize/2; mask > 0; mask /= 2) 
            {
                sqSum += __shfl_xor(sqSum, mask);
            }
            if(tWarpIdx==0)
            {
                atomicAdd(&std_cache[0],sqSum);
            }
            __syncthreads();
            if(tSliceIdx==0)
            {
                std_cache[0] = rsqrt(std_cache[0]*i_n+epsilon);
            }
            __syncthreads();
            rstd = std_cache[0];
            // if(threadIdx.x==0&&blockIdx.x==0)printf("rstd:%f\n", rstd);

            UNROLL for(int m=0;m<mult;m++)
            {
                if(tSliceIdx+m*blockDim.x<in_depth&&thread_id[m]<n_inputs)
                {
                  // const T tmp_out = (inp[m]-mu)*rstd;
                  // if(threadIdx.x==0&&blockIdx.x==0)printf("m:%d,o:%f\n",m,tmp_out);
                  output[thread_id[m]] = (inp[m]-mu)*rstd*_gamma[m]+_beta[m];
                  // output[thread_id[m]] = tmp_out;
                }
            }
            __syncthreads();
        }
    }

    //fused small LN kernel
    template<typename T>
    __global__ void LayerNormFusedSmallGPUKernel(const LayerNormFusedArgs args,
      const T* __restrict__ input,
      const T* __restrict__ gamma,
      const T* __restrict__ beta,
        T* __restrict__ output,
        const int num_blocks,const int slice_per_block
        )
    {
        const int slice_size = args.slice_size;
        const int in_depth = args.depth;
        const int n_inputs = args.n_inputs;
        const T epsilon = args.epsilon;

        const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

        const int slice_id = threadIdx.x/slice_size;
        const int tSliceIdx = threadIdx.x%slice_size;

        const T _gamma = get_value<T>(gamma+tSliceIdx,tSliceIdx,in_depth);
        const T _beta = get_value<T>(beta+tSliceIdx,tSliceIdx,in_depth);

        T mu;
        T rstd;
        
        for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
        {   
          mu = static_cast<T>(0.0f);
          rstd = static_cast<T>(0.0f);

          const int thread_id = (bId*slice_per_block+slice_id)*in_depth+tSliceIdx;
          // const T inp = 0;
          const T inp = get_value<T>(input+thread_id,tSliceIdx,in_depth);

          mu += inp*i_n;

          for (int mask = slice_size/2; mask > 0; mask /= 2) 
          {
              mu +=__shfl_xor(mu, mask);
          }

          if (tSliceIdx<in_depth) rstd += (inp-mu)*(inp-mu);
          // rstd += (inp-mu)*(inp-mu);

          for (int mask = slice_size/2; mask > 0; mask /= 2) 
          {
              rstd += __shfl_xor(rstd, mask);
          }

          rstd = rsqrt(rstd*i_n+epsilon);

          if (tSliceIdx<in_depth&& thread_id<n_inputs) output[thread_id] = (inp-mu)*rstd*_gamma+_beta;
        }
    }
}  // namespace
#define LN_GPU_KERNEL(mult) LayerNormGPUKernel<T,mult><<<  \
                grid_size,   \
                block_size,  \
                sbytes      \
                >>>\
                (args, input, output,num_blocks)
// A simple launch pad to launch the Cuda kernel for Layer Normalization.
template <typename T>
struct LayerNormGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args, const T* input,
                  T* output) {
    const int warp_size = 32;
    if(args.slice_size<=warp_size)
    {
      const int block_size = 256;
      const int slice_per_block = block_size/args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices,slice_per_block);
      const int grid_size = std::min(120,num_blocks);
      LayerNormSmallGPUKernel<T><<<
                grid_size,
                block_size,
                0
                >>>(
                args, input, output,
                num_blocks,slice_per_block);
    }else{
      // limit the numebr of threads per block to reduce performance hit on __syncthreads.
      int block_size;
      int mult;
      get_block_size(args.slice_size,block_size,mult);
      const int num_blocks = args.n_slices;
      const int sbytes = 2*sizeof(T);
      const int grid_size = std::min(MAX_GRID_SIZE,num_blocks);
      switch(mult)
      {
        case 1: LN_GPU_KERNEL(1);
                break;
        case 2: LN_GPU_KERNEL(2);
                break;
        case 3: LN_GPU_KERNEL(3);
                break;
        case 4: LN_GPU_KERNEL(4);
                break;
        case 5: LN_GPU_KERNEL(5);
                break;
      }
  }
}
};

template struct LayerNormGPULaunch<float>;
template struct LayerNormGPULaunch<double>;


template<typename T, int mult>
__global__ void LayerNormBackpropGPUKernel(const LayerNormFusedArgs args,
  const T* __restrict__ input,
  const T* __restrict__ out_back,
    T* __restrict__ in_back,
    const int num_blocks
    )
{
    const int in_depth = args.depth;
    const int slice_size = args.slice_size;
    const int n_inputs = args.n_inputs;
    const T epsilon = args.epsilon;

    const int tSliceIdx = threadIdx.x%slice_size;
    const int tWarpIdx = threadIdx.x%warpSize;

    const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T* mean_cache = (T*)my_smem;
    T* std_cache = (T*)&my_smem[sizeof(T)];
    T* dmu_cache = (T*)&my_smem[2*sizeof(T)];
    T* dstd_cache = (T*)&my_smem[3*sizeof(T)];

    T inp[mult];
    T dout[mult];

    int thread_id[mult];

    T mu;
    T rstd;
    T dstd;
    T dmu;

    for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
    {   
        mu = static_cast<T>(0.0f);
        rstd = static_cast<T>(0.0f);
        dmu = static_cast<T>(0.0f);
        dstd = static_cast<T>(0.0f);

        if(tSliceIdx==0){
          mean_cache[0]=static_cast<T>(0.0f);
          std_cache[0]=static_cast<T>(0.0f);
          dmu_cache[0]=static_cast<T>(0.0f);
          dstd_cache[0]=static_cast<T>(0.0f);
        }
        __syncthreads();
        UNROLL for(int m=0;m<mult;m++)
        {
            thread_id[m] = bId*in_depth+m*blockDim.x+tSliceIdx;
        }
        UNROLL for(int m=0;m<mult;m++)
        {
            inp[m] = get_value<T>(input+thread_id[m],tSliceIdx+m*blockDim.x,in_depth);
            dout[m] = get_value<T>(out_back+thread_id[m],tSliceIdx+m*blockDim.x,in_depth);
        }

        UNROLL for(int m=0;m<mult;m++)
        {
            mu += inp[m]*i_n;
            dmu += dout[m]*i_n;
        }   

        warpSum<T>(mu,dmu);
        if(tWarpIdx==0)
        {
            atomicAdd(&mean_cache[0],mu);
            atomicAdd(&dmu_cache[0],dmu);
        }
        __syncthreads();

        mu = mean_cache[0];
        UNROLL for(int m=0;m<mult;m++)
        {
          if (tSliceIdx+m*blockDim.x<in_depth)
          {
            rstd += (inp[m]-mu)*(inp[m]-mu);
            dstd += (inp[m]-mu)*dout[m];
          }
        }   

        warpSum<T>(rstd,dstd);

        if(tWarpIdx==0)
        {
            atomicAdd(&std_cache[0],rstd);
            atomicAdd(&dstd_cache[0],dstd);
        }
        __syncthreads();
        if(tSliceIdx==0)
        {
            rstd = rsqrt(std_cache[0]*i_n+epsilon);
            std_cache[0] = rstd;
            dmu_cache[0] = dmu_cache[0] *rstd;
            dstd_cache[0] = dstd_cache[0]*rstd*rstd*rstd*i_n;
        }
        __syncthreads();
        rstd = std_cache[0];
        dstd = dstd_cache[0];
        dmu = dmu_cache[0];


        UNROLL for(int m=0;m<mult;m++)
        {
          if(tSliceIdx+m*blockDim.x<in_depth&&thread_id[m]<n_inputs)
          {
            in_back[thread_id[m]] = dout[m]*rstd-(inp[m]-mu)*dstd-dmu;
          }
        }
        __syncthreads();
    }
}


template<typename T>
__global__ void LayerNormSmallBackpropGPUKernel(const LayerNormFusedArgs args,
  const T* __restrict__ input,
  const T* __restrict__ out_back,
    T* __restrict__ in_back,
    const int num_blocks,const int slice_per_block
    )
{

    const int in_depth = args.depth;
    const int slice_size = args.slice_size;
    const int n_inputs = args.n_inputs;
    const T epsilon = args.epsilon;

    const int slice_id = threadIdx.x/slice_size;
    const int tSliceIdx = threadIdx.x%slice_size;

    const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

    T mu;
    T rstd;
    T dstd;
    T dmu;

    // we need a thread block here to ensure initialization is complete
    __syncthreads();
    for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
    {   
        mu = static_cast<T>(0.0f);
        rstd = static_cast<T>(0.0f);
        dmu = static_cast<T>(0.0f);
        dstd = static_cast<T>(0.0f);


        const int thread_id = (bId*slice_per_block+slice_id)*in_depth+tSliceIdx;
        const T inp = get_value<T>(input+thread_id,tSliceIdx,in_depth);
        const T dout = get_value<T>(out_back+thread_id,tSliceIdx,in_depth);

        mu += inp*i_n;
        dmu += dout*i_n;

        for (int mask = slice_size/2; mask > 0; mask /= 2) 
        {
            mu +=__shfl_xor(mu, mask);
            dmu +=__shfl_xor(dmu, mask);
        }

        if (tSliceIdx<in_depth)
        {
          rstd += (inp-mu)*(inp-mu);
          dstd += (inp-mu)*dout;
        }

        for (int mask = slice_size/2; mask > 0; mask /= 2) 
        {
            rstd += __shfl_xor(rstd, mask);
            dstd += __shfl_xor(dstd, mask);
        }

        rstd = rsqrt(rstd*i_n+epsilon);
        dmu = dmu *rstd;
        dstd = dstd*rstd*rstd*rstd*i_n;

        if (tSliceIdx<in_depth&& thread_id<n_inputs) in_back[thread_id] = dout*rstd-(inp-mu)*dstd-dmu;
    }
}

#define LN_GPU_BACKPROP_KERNEL(mult) LayerNormBackpropGPUKernel<T,mult><<<  \
                grid_size,   \
                block_size,  \
                sbytes      \
                >>>\
                (args,input,out_back,in_back,num_blocks)
template <typename T>
struct LayerNormBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input,const T* out_back,
                  T* in_back) {
    const int warp_size = 32;
    if(args.slice_size<=warp_size)
    {
      const int block_size = 128;
      const int slice_per_block = block_size/args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices,slice_per_block);
      const int grid_size = std::min(120,num_blocks);
      const int sbytes = 0;
      // printf("slice_per_block:%d,grid_size:%d\n",slice_per_block,grid_size);
      LayerNormSmallBackpropGPUKernel<T><<<
                grid_size,
                block_size,
                sbytes
                >>>(
                args,input, out_back,in_back,
                num_blocks,slice_per_block);
    }else{
      // limit the numebr of threads per block to reduce performance hit on __syncthreads.
      int block_size;
      int mult;
      get_block_size(args.slice_size,block_size,mult);
      const int num_blocks = args.n_slices;
      const int sbytes = 4*sizeof(T);
      const int max_grid = args.n_slices<2*MAX_GRID_SIZE?60:MAX_GRID_SIZE;
      const int grid_size = std::min(max_grid,num_blocks);
      switch(mult)
      {
        case 1: LN_GPU_BACKPROP_KERNEL(1);
                break;
        case 2: LN_GPU_BACKPROP_KERNEL(2);
                break;
        case 3: LN_GPU_BACKPROP_KERNEL(3);
                break;
        case 4: LN_GPU_BACKPROP_KERNEL(4);
                break;
        case 5: LN_GPU_BACKPROP_KERNEL(5);
                break;
      }
    }
  }
};

template struct LayerNormBackpropGPULaunch<float>;
template struct LayerNormBackpropGPULaunch<double>;

#define LN_GPU_BIASADD_KERNEL(mult) LayerNormBiasAddGPUKernel<T,mult><<<  \
                grid_size,   \
                block_size,  \
                sbytes      \
                >>>\
                (args, input,beta, output,num_blocks)
// A simple launch pad to launch the Cuda kernel for Layer Normalization.
template <typename T>
struct LayerNormBiasAddGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args, const T* input,
                  const T* beta,T* output) {
    const int warp_size = 32;
    if(args.slice_size<=warp_size)
    {
      const int block_size = 256;
      const int slice_per_block = block_size/args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices,slice_per_block);
      const int grid_size = std::min(120,num_blocks);
      LayerNormBiasAddSmallGPUKernel<T><<<
                grid_size,
                block_size,
                0
                >>>(
                args, input,beta, output,
                num_blocks,slice_per_block);
    }else{
      // limit the numebr of threads per block to reduce performance hit on __syncthreads.
      int block_size;
      int mult;
      get_block_size(args.slice_size,block_size,mult);
      const int num_blocks = args.n_slices;
      const int sbytes = 2*sizeof(T);
      const int grid_size = std::min(MAX_GRID_SIZE,num_blocks);
      switch(mult)
      {
        case 1: LN_GPU_BIASADD_KERNEL(1);
                break;
        case 2: LN_GPU_BIASADD_KERNEL(2);
                break;
        case 3: LN_GPU_BIASADD_KERNEL(3);
                break;
        case 4: LN_GPU_BIASADD_KERNEL(4);
                break;
        case 5: LN_GPU_BIASADD_KERNEL(5);
                break;
      }
  }
}
};

template struct LayerNormBiasAddGPULaunch<float>;
template struct LayerNormBiasAddGPULaunch<double>;


template<typename T, int mult>
__global__ void LayerNormBiasAddBackpropGPUKernel(const LayerNormFusedArgs args,
  const T* __restrict__ input,
  const T* __restrict__ out_back,
    T* __restrict__ in_back,
    T* __restrict__ beta_back,
    const int num_blocks
    )
{
    const int in_depth = args.depth;
    const int slice_size = args.slice_size;
    const int n_inputs = args.n_inputs;
    const T epsilon = args.epsilon;

    const int tSliceIdx = threadIdx.x%slice_size;
    const int tWarpIdx = threadIdx.x%warpSize;

    const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T* mean_cache = (T*)my_smem;
    T* std_cache = (T*)&my_smem[sizeof(T)];
    T* dmu_cache = (T*)&my_smem[2*sizeof(T)];
    T* dstd_cache = (T*)&my_smem[3*sizeof(T)];

    T inp[mult];
    T dout[mult];

    T _beta_bp[mult];
    int thread_id[mult];

    UNROLL for(int m=0;m<mult;m++)
    {
      _beta_bp[m] = static_cast<T>(0.0f);
    }

    T mu;
    T rstd;
    T dstd;
    T dmu;

    for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
    {   
        mu = static_cast<T>(0.0f);
        rstd = static_cast<T>(0.0f);
        dmu = static_cast<T>(0.0f);
        dstd = static_cast<T>(0.0f);

        if(tSliceIdx==0){
          mean_cache[0]=static_cast<T>(0.0f);
          std_cache[0]=static_cast<T>(0.0f);
          dmu_cache[0]=static_cast<T>(0.0f);
          dstd_cache[0]=static_cast<T>(0.0f);
        }
        __syncthreads();
        UNROLL for(int m=0;m<mult;m++)
        {
            thread_id[m] = bId*in_depth+m*blockDim.x+tSliceIdx;
        }
        UNROLL for(int m=0;m<mult;m++)
        {
            inp[m] = get_value<T>(input+thread_id[m],tSliceIdx+m*blockDim.x,in_depth);
            dout[m] = get_value<T>(out_back+thread_id[m],tSliceIdx+m*blockDim.x,in_depth);
        }

        UNROLL for(int m=0;m<mult;m++)
        {
            _beta_bp[m] += dout[m];
            mu += inp[m]*i_n;
            dmu += dout[m]*i_n;
        }   

        warpSum<T>(mu,dmu);
        if(tWarpIdx==0)
        {
            atomicAdd(&mean_cache[0],mu);
            atomicAdd(&dmu_cache[0],dmu);
        }
        __syncthreads();

        mu = mean_cache[0];
        UNROLL for(int m=0;m<mult;m++)
        {
          if (tSliceIdx+m*blockDim.x<in_depth)
          {
            rstd += (inp[m]-mu)*(inp[m]-mu);
            dstd += (inp[m]-mu)*dout[m];
          }
        }   

        warpSum<T>(rstd,dstd);

        if(tWarpIdx==0)
        {
            atomicAdd(&std_cache[0],rstd);
            atomicAdd(&dstd_cache[0],dstd);
        }
        __syncthreads();
        if(tSliceIdx==0)
        {
            rstd = rsqrt(std_cache[0]*i_n+epsilon);
            std_cache[0] = rstd;
            dmu_cache[0] = dmu_cache[0] *rstd;
            dstd_cache[0] = dstd_cache[0]*rstd*rstd*rstd*i_n;
        }
        __syncthreads();
        rstd = std_cache[0];
        dstd = dstd_cache[0];
        dmu = dmu_cache[0];


        UNROLL for(int m=0;m<mult;m++)
        {
          if(tSliceIdx+m*blockDim.x<in_depth&&thread_id[m]<n_inputs)
          {
            in_back[thread_id[m]] = dout[m]*rstd-(inp[m]-mu)*dstd-dmu;
          }
        }
        __syncthreads();
    }
    UNROLL for(int m=0;m<mult;m++)
    {
      if(tSliceIdx+m*blockDim.x<in_depth)
      {
        atomicAdd(beta_back+tSliceIdx+m*blockDim.x,_beta_bp[m]);
      }
    }
}


template<typename T>
__global__ void LayerNormBiasAddSmallBackpropGPUKernel(const LayerNormFusedArgs args,
  const T* __restrict__ input,
  const T* __restrict__ out_back,
    T* __restrict__ in_back,
    T* __restrict__ beta_back,
    const int num_blocks,const int slice_per_block
    )
{

    const int in_depth = args.depth;
    const int slice_size = args.slice_size;
    const int n_inputs = args.n_inputs;
    const T epsilon = args.epsilon;

    const int slice_id = threadIdx.x/slice_size;
    const int tSliceIdx = threadIdx.x%slice_size;
    const int tWarpIdx = threadIdx.x%warpSize;

    const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T* beta_cache = (T*)&my_smem[in_depth*sizeof(T)];
    //initialize shared memory cache to 0.0
    if(threadIdx.x<in_depth)
    {
      beta_cache[threadIdx.x] = static_cast<T>(0.0f);
    }

    T mu;
    T rstd;
    T dstd;
    T dmu;

    T _beta_bp=static_cast<T>(0.0f);
    // we need a thread block here to ensure initialization is complete
    __syncthreads();
    for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
    {   
        mu = static_cast<T>(0.0f);
        rstd = static_cast<T>(0.0f);
        dmu = static_cast<T>(0.0f);
        dstd = static_cast<T>(0.0f);


        const int thread_id = (bId*slice_per_block+slice_id)*in_depth+tSliceIdx;
        const T inp = get_value<T>(input+thread_id,tSliceIdx,in_depth);
        const T dout = get_value<T>(out_back+thread_id,tSliceIdx,in_depth);

        _beta_bp += dout;
        mu += inp*i_n;
        dmu += dout*i_n;

        for (int mask = slice_size/2; mask > 0; mask /= 2) 
        {
            mu +=__shfl_xor(mu, mask);
            dmu +=__shfl_xor(dmu, mask);
        }

        if (tSliceIdx<in_depth)
        {
          rstd += (inp-mu)*(inp-mu);
          dstd += (inp-mu)*dout;
        }

        for (int mask = slice_size/2; mask > 0; mask /= 2) 
        {
            rstd += __shfl_xor(rstd, mask);
            dstd += __shfl_xor(dstd, mask);
        }

        rstd = rsqrt(rstd*i_n+epsilon);
        dmu = dmu *rstd;
        dstd = dstd*rstd*rstd*rstd*i_n;

        if (tSliceIdx<in_depth&& thread_id<n_inputs) in_back[thread_id] = dout*rstd-(inp-mu)*dstd-dmu;
    }
    for (int mask = slice_size; mask < warpSize; mask *= 2)
    {
      _beta_bp += __shfl_xor(_beta_bp, mask);
    }
    //accumulate *_bp into shared memory.
    if(tWarpIdx<in_depth)
    {
      atomicAdd(beta_cache+tSliceIdx,_beta_bp);
    }
    //add *_bp into global memory.
    __syncthreads();
    if(slice_id==0 && tSliceIdx<in_depth)
    {
      atomicAdd(beta_back+tSliceIdx,beta_cache[tSliceIdx]);
    }
}

template<typename T>
__global__ void initialize_beta_with_zeros(T* beta_back,const int n_inputs)
{
  const int thread_id = threadIdx.x+blockDim.x*blockIdx.x;
  if(thread_id<n_inputs)
  {
    beta_back[thread_id] = static_cast<T>(0.0f);
  }
}

template<typename T>
void initialize_beta(const LayerNormFusedArgs args,T* beta_back)
{
  const int block_size = std::min(args.depth,256);
  const int grid_size = get_num_blocks(args.depth,block_size);
  initialize_beta_with_zeros<T><<<grid_size,block_size>>>(beta_back,args.depth);
}
#define LN_GPU_BIASADD_BACKPROP_KERNEL(mult) LayerNormBiasAddBackpropGPUKernel<T,mult><<<  \
                grid_size,   \
                block_size,  \
                sbytes      \
                >>>\
                (args,input,out_back,in_back,beta_back,num_blocks)
template <typename T>
struct LayerNormBiasAddBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input,const T* out_back,
                  T* in_back,T* beta_back) {
    const int warp_size = 32;
    initialize_beta<T>(args,beta_back);
    if(args.slice_size<=warp_size)
    {
      const int block_size = 128;
      const int slice_per_block = block_size/args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices,slice_per_block);
      const int grid_size = std::min(120,num_blocks);
      const int sbytes = (2*args.depth)*sizeof(T);
      // printf("slice_per_block:%d,grid_size:%d\n",slice_per_block,grid_size);
      LayerNormBiasAddSmallBackpropGPUKernel<T><<<
                grid_size,
                block_size,
                sbytes
                >>>(
                args,input, out_back,in_back,beta_back,
                num_blocks,slice_per_block);
    }else{
      // limit the numebr of threads per block to reduce performance hit on __syncthreads.
      int block_size;
      int mult;
      get_block_size(args.slice_size,block_size,mult);
      const int num_blocks = args.n_slices;
      const int sbytes = 4*sizeof(T);
      const int max_grid = args.n_slices<2*MAX_GRID_SIZE?60:MAX_GRID_SIZE;
      const int grid_size = std::min(max_grid,num_blocks);
      switch(mult)
      {
        case 1: LN_GPU_BIASADD_BACKPROP_KERNEL(1);
                break;
        case 2: LN_GPU_BIASADD_BACKPROP_KERNEL(2);
                break;
        case 3: LN_GPU_BIASADD_BACKPROP_KERNEL(3);
                break;
        case 4: LN_GPU_BIASADD_BACKPROP_KERNEL(4);
                break;
        case 5: LN_GPU_BIASADD_BACKPROP_KERNEL(5);
                break;
      }
    }
  }
};

template struct LayerNormBiasAddBackpropGPULaunch<float>;
template struct LayerNormBiasAddBackpropGPULaunch<double>;

#define LN_GPU_FUSED_KERNEL(mult) LayerNormFusedGPUKernel<T,mult><<<  \
                grid_size,   \
                block_size,  \
                sbytes      \
                >>>\
                (args, input,gamma,beta, output,num_blocks)
// A simple launch pad to launch the Cuda kernel for Layer Normalization.
template <typename T>
struct LayerNormFusedGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args, const T* input,
                  const T* gamma,const T* beta,T* output) {
    const int warp_size = 32;
    // fillZeros<T><<<60,256>>>(output,args.n_inputs);

    if(args.slice_size<=warp_size)
    {
      const int block_size = 256;
      const int slice_per_block = block_size/args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices,slice_per_block);
      const int grid_size = std::min(120,num_blocks);
      // printf("slice_per_block:%d,grid_size:%d\n",slice_per_block,grid_size);
      LayerNormFusedSmallGPUKernel<T><<<
                grid_size,
                block_size,
                0
                >>>(
                args, input,gamma,beta, output,
                num_blocks,slice_per_block);
    }else{
      // limit the numebr of threads per block to reduce performance hit on __syncthreads.
      int block_size;
      int mult;
      get_block_size(args.slice_size,block_size,mult);
      const int num_blocks = args.n_slices;
      const int sbytes = 2*sizeof(T);
      // printf("mult:%d,bs:%d,nb:%d,spb:%d\n",mult,block_size,num_blocks,slice_per_block);
      const int grid_size = std::min(MAX_GRID_SIZE,num_blocks);
      switch(mult)
      {
        case 1: LN_GPU_FUSED_KERNEL(1);
                break;
        case 2: LN_GPU_FUSED_KERNEL(2);
                break;
        case 3: LN_GPU_FUSED_KERNEL(3);
                break;
        case 4: LN_GPU_FUSED_KERNEL(4);
                break;
        case 5: LN_GPU_FUSED_KERNEL(5);
                break;
      }
  }
}
};

template struct LayerNormFusedGPULaunch<float>;
template struct LayerNormFusedGPULaunch<double>;


template<typename T, int mult>
__global__ void LayerNormFusedBackpropGPUKernel(const LayerNormFusedArgs args,
  const T* __restrict__ input,
  const T* __restrict__ out_back,
  const T* __restrict__ gamma,
    T* __restrict__ in_back,
    T* __restrict__ gamma_back,
    T* __restrict__ beta_back,
    const int num_blocks
    )
{
    const int in_depth = args.depth;
    const int n_inputs = args.n_inputs;
    const T epsilon = args.epsilon;

    const int tWarpIdx = threadIdx.x%warpSize;

    const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T* mean_cache = (T*)my_smem;
    T* std_cache = (T*)&my_smem[sizeof(T)];
    T* dmu_cache = (T*)&my_smem[2*sizeof(T)];
    T* dstd_cache = (T*)&my_smem[3*sizeof(T)];

    T inp[mult];
    T dout[mult];

    T _gamma[mult];
    T _gamma_bp[mult];
    T _beta_bp[mult];
    int thread_id[mult];

    UNROLL for(int m=0;m<mult;m++)
    {
      _beta_bp[m] = static_cast<T>(0.0f);
      _gamma_bp[m] = static_cast<T>(0.0f);
      _gamma[m] = get_value<T>(gamma+threadIdx.x+m*blockDim.x,threadIdx.x+m*blockDim.x,in_depth);
    }

    T mu;
    T rstd;
    T dstd;
    T dmu;
    for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
    {   
        mu = static_cast<T>(0.0f);
        rstd = static_cast<T>(0.0f);
        dmu = static_cast<T>(0.0f);
        dstd = static_cast<T>(0.0f);

        if(threadIdx.x==0){
          mean_cache[0]=static_cast<T>(0.0f);
          std_cache[0]=static_cast<T>(0.0f);
          dmu_cache[0]=static_cast<T>(0.0f);
          dstd_cache[0]=static_cast<T>(0.0f);
        }
        __syncthreads();
        UNROLL for(int m=0;m<mult;m++)
        {
            thread_id[m] = bId*in_depth+m*blockDim.x+threadIdx.x;
        }
        UNROLL for(int m=0;m<mult;m++)
        {
            inp[m] = get_value<T>(input+thread_id[m],threadIdx.x+m*blockDim.x,in_depth);
            dout[m] = get_value<T>(out_back+thread_id[m],threadIdx.x+m*blockDim.x,in_depth);
        }

        UNROLL for(int m=0;m<mult;m++)
        {
            _beta_bp[m] += dout[m];
            mu += inp[m]*i_n;
            dmu += dout[m]*_gamma[m]*i_n;
        }   

        warpSum<T>(mu,dmu);
        if(tWarpIdx==0)
        {
            atomicAdd(&mean_cache[0],mu);
            atomicAdd(&dmu_cache[0],dmu);
        }
        __syncthreads();

        mu = mean_cache[0];
        UNROLL for(int m=0;m<mult;m++)
        {
          if (threadIdx.x+m*blockDim.x<in_depth)
          {
            rstd += (inp[m]-mu)*(inp[m]-mu);
            dstd += (inp[m]-mu)*dout[m]*_gamma[m];
          }
        }   

        warpSum<T>(rstd,dstd);

        if(tWarpIdx==0)
        {
            atomicAdd(&std_cache[0],rstd);
            atomicAdd(&dstd_cache[0],dstd);
        }
        __syncthreads();
        if(threadIdx.x==0)
        {
            rstd = rsqrt(std_cache[0]*i_n+epsilon);
            std_cache[0] = rstd;
            dmu_cache[0] = dmu_cache[0] *rstd;
            dstd_cache[0] = dstd_cache[0]*rstd*rstd*rstd*i_n;
        }
        __syncthreads();
        rstd = std_cache[0];
        dstd = dstd_cache[0];
        dmu = dmu_cache[0];


        UNROLL for(int m=0;m<mult;m++)
        {
          if(threadIdx.x+m*blockDim.x<in_depth&&thread_id[m]<n_inputs)
          {
            _gamma_bp[m] += dout[m]*(inp[m]-mu)*rstd;
            in_back[thread_id[m]] = dout[m]*_gamma[m]*rstd-(inp[m]-mu)*dstd-dmu;
          }
        }
        __syncthreads();
    }

    UNROLL for(int m=0;m<mult;m++)
    {
      if(threadIdx.x+m*blockDim.x<in_depth)
      {
        atomicAdd(gamma_back+threadIdx.x+m*blockDim.x,_gamma_bp[m]);
        atomicAdd(beta_back+threadIdx.x+m*blockDim.x,_beta_bp[m]);
      }
    }
}


template<typename T>
__global__ void LayerNormFusedSmallBackpropGPUKernel(const LayerNormFusedArgs args,
  const T* __restrict__ input,
  const T* __restrict__ out_back,
  const T* __restrict__ gamma,
    T* __restrict__ in_back,
    T* __restrict__ gamma_back,
    T* __restrict__ beta_back,
    const int num_blocks,const int slice_per_block
    )
{

    const int in_depth = args.depth;
    const int slice_size = args.slice_size;
    const int n_inputs = args.n_inputs;
    const T epsilon = args.epsilon;

    const int slice_id = threadIdx.x/slice_size;
    const int tSliceIdx = threadIdx.x%slice_size;
    const int tWarpIdx = threadIdx.x%warpSize;

    const T i_n = static_cast<T>(1.0f)/static_cast<T>(in_depth);

    extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
    T* gamma_cache = (T*)my_smem;
    T* beta_cache = (T*)&my_smem[in_depth*sizeof(T)];
    //initialize shared memory cache to 0.0
    if(threadIdx.x<in_depth)
    {
      gamma_cache[threadIdx.x] = static_cast<T>(0.0f);
      beta_cache[threadIdx.x] = static_cast<T>(0.0f);
    }

    const T _gamma = get_value<T>(gamma+tSliceIdx,tSliceIdx,in_depth);
    T mu;
    T rstd;
    T dstd;
    T dmu;

    T _gamma_bp=static_cast<T>(0.0f);
    T _beta_bp=static_cast<T>(0.0f);
    // we need a thread block here to ensure initialization is complete
    __syncthreads();
    for(int bId=blockIdx.x;bId<num_blocks;bId+=gridDim.x)
    {   
        mu = static_cast<T>(0.0f);
        rstd = static_cast<T>(0.0f);
        dmu = static_cast<T>(0.0f);
        dstd = static_cast<T>(0.0f);


        const int thread_id = (bId*slice_per_block+slice_id)*in_depth+tSliceIdx;
        const T inp = get_value<T>(input+thread_id,tSliceIdx,in_depth);
        const T dout = get_value<T>(out_back+thread_id,tSliceIdx,in_depth);

        const T dout_g = dout*_gamma;
        _beta_bp += dout;
        mu += inp*i_n;
        dmu += dout*_gamma*i_n;

        for (int mask = slice_size/2; mask > 0; mask /= 2) 
        {
            mu +=__shfl_xor(mu, mask);
            dmu +=__shfl_xor(dmu, mask);
        }

        if (tSliceIdx<in_depth)
        {
          rstd += (inp-mu)*(inp-mu);
          dstd += (inp-mu)*dout_g;
        }

        for (int mask = slice_size/2; mask > 0; mask /= 2) 
        {
            rstd += __shfl_xor(rstd, mask);
            dstd += __shfl_xor(dstd, mask);
        }

        rstd = rsqrt(rstd*i_n+epsilon);
        dmu = dmu *rstd;
        dstd = dstd*rstd*rstd*rstd*i_n;

        if (tSliceIdx<in_depth&& thread_id<n_inputs)
        {
          _gamma_bp += dout*(inp-mu)*rstd;
          in_back[thread_id] = dout_g*rstd-(inp-mu)*dstd-dmu;
        } 
    }
    for (int mask = slice_size; mask < warpSize; mask *= 2)
    {
      _gamma_bp += __shfl_xor(_gamma_bp, mask);
      _beta_bp += __shfl_xor(_beta_bp, mask);
    }
    //accumulate *_bp into shared memory.
    if(tWarpIdx<in_depth)
    {
      atomicAdd(gamma_cache+tSliceIdx,_gamma_bp);
      atomicAdd(beta_cache+tSliceIdx,_beta_bp);
    }
    //add *_bp into global memory.
    __syncthreads();
    if(slice_id==0 && tSliceIdx<in_depth)
    {
      atomicAdd(gamma_back+tSliceIdx,gamma_cache[tSliceIdx]);
      atomicAdd(beta_back+tSliceIdx,beta_cache[tSliceIdx]);
    }
}

template<typename T>
__global__ void initialize_with_zeros(T* gamma_back,T* beta_back,const int n_inputs)
{
  const int thread_id = threadIdx.x+blockDim.x*(blockIdx.x/2);
  if(thread_id<n_inputs)
  {
    if(blockIdx.x%2==0){
      gamma_back[thread_id] = static_cast<T>(0.0f);
    }else
      beta_back[thread_id] = static_cast<T>(0.0f);
  }
}

template<typename T>
void initialize_outputs(const LayerNormFusedArgs args,T* gamma_back,T* beta_back)
{
  const int block_size = std::min(args.depth,256);
  const int grid_size = get_num_blocks(args.depth,block_size)*2;
  initialize_with_zeros<T><<<grid_size,block_size>>>(gamma_back,beta_back,args.depth);
}
#define LN_GPU_FUSED_BACKPROP_KERNEL(mult) LayerNormFusedBackpropGPUKernel<T,mult><<<  \
                grid_size,   \
                block_size,  \
                sbytes      \
                >>>\
                (args,input,out_back,gamma, in_back,gamma_back,beta_back,num_blocks)
template <typename T>
struct LayerNormFusedBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input,const T* out_back,const T* gamma,
                  T* in_back,T* gamma_back,T* beta_back) {
    const int warp_size = 32;
    initialize_outputs<T>(args,gamma_back,beta_back);
    // printf("inp:%p,ob:%p,gam:%p,ib:%p,gb:%p,bb:%p\n", input,out_back,gamma,in_back,gamma_back,beta_back);
    if(args.slice_size<=warp_size)
    {
      const int block_size = 128;
      const int slice_per_block = block_size/args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices,slice_per_block);
      const int grid_size = std::min(120,num_blocks);
      const int sbytes = (2*args.depth)*sizeof(T);
      // printf("slice_per_block:%d,grid_size:%d\n",slice_per_block,grid_size);
      LayerNormFusedSmallBackpropGPUKernel<T><<<
                grid_size,
                block_size,
                sbytes
                >>>(
                args,input, out_back,gamma, in_back,gamma_back,beta_back,
                num_blocks,slice_per_block);
    }else{
      // limit the numebr of threads per block to reduce performance hit on __syncthreads.
      int block_size;
      int mult;
      get_block_size(args.slice_size,block_size,mult);
      const int num_blocks = args.n_slices;
      const int sbytes = 4*sizeof(T);
      const int max_grid = args.n_slices<2*MAX_GRID_SIZE?60:MAX_GRID_SIZE;
      const int grid_size = std::min(max_grid,num_blocks);
      switch(mult)
      {
        case 1: LN_GPU_FUSED_BACKPROP_KERNEL(1);
                break;
        case 2: LN_GPU_FUSED_BACKPROP_KERNEL(2);
                break;
        case 3: LN_GPU_FUSED_BACKPROP_KERNEL(3);
                break;
        case 4: LN_GPU_FUSED_BACKPROP_KERNEL(4);
                break;
        case 5: LN_GPU_FUSED_BACKPROP_KERNEL(5);
                break;
      }
    }
  }
};

template struct LayerNormFusedBackpropGPULaunch<float>;
template struct LayerNormFusedBackpropGPULaunch<double>;
}  // namespace tensorflow
