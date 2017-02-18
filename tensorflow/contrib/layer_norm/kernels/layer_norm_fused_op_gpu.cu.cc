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
#include <math.h>
#include <stdio.h>
#include <algorithm>

#if GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "./layer_norm_fused_op.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if !defined(_MSC_VER)
#define UNROLL _Pragma("unroll")
#else
#define UNROLL
#endif

template <typename T>
__device__ __host__ inline T ldg(const T* address) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* a, double b) { return b; }
#endif

const static int MAX_GRID_SIZE = 480;
const static int MAX_N_ILP = 5;
const static int WARP_SIZE = 32;

inline int get_num_blocks(const int n_slices, const int slice_per_block) {
  const int num_blocks = n_slices / slice_per_block;
  if (num_blocks * slice_per_block == n_slices)
    return num_blocks;
  else
    return num_blocks + 1;
}

// Calculates optimal block_size and n_ILP(number of instruction level
// parallelism)
inline int get_block_size(const int slice_size, int& block_size, int& n_ILP) {
  int tmp_block_size = WARP_SIZE;
  int tmp_n_ILP = slice_size / tmp_block_size;
  n_ILP =
      (tmp_n_ILP * tmp_block_size) >= slice_size ? tmp_n_ILP : (tmp_n_ILP + 1);
  while (n_ILP > MAX_N_ILP) {
    tmp_block_size += WARP_SIZE;
    tmp_n_ILP = slice_size / tmp_block_size;
    n_ILP = (tmp_n_ILP * tmp_block_size) >= slice_size ? tmp_n_ILP
                                                       : (tmp_n_ILP + 1);
  }
  block_size = tmp_block_size;
  return 0;
}

template <typename T>
__device__ __inline__ void warpSum(T& val1, T& val2) {
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
__device__ __inline__ T get_value(const T* address, const int bound_check,
                                  const int up_bound) {
  if (bound_check < up_bound)
    return ldg(address);
  else
    return static_cast<T>(0.0f);
}

namespace tensorflow {

namespace {

typedef Eigen::GpuDevice GPUDevice;

template <typename T, int n_ILP>
__global__ void LayerNormGPUKernel(const LayerNormFusedArgs args,
                                   const T* __restrict__ input,
                                   T* __restrict__ output,
                                   const int num_blocks) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int lane_id = threadIdx.x % warpSize;
  const int thread_slice_id = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* mean_cache = (T*)my_smem;
  T* std_cache = mean_cache + 1;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);
  T inp[n_ILP];
  int thread_id[n_ILP];

  T sum;
  T sqSum;
  T mu;
  T rstd;

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = static_cast<T>(0.0f);
    sqSum = static_cast<T>(0.0f);

    if (thread_slice_id == 0) {
      *mean_cache = static_cast<T>(0.0f);
      *std_cache = static_cast<T>(0.0f);
    }
    __syncthreads();

    UNROLL for (int m = 0; m < n_ILP; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + thread_slice_id;
    }

    UNROLL for (int m = 0; m < n_ILP; m++) {
      inp[m] = get_value<T>(input + thread_id[m],
                            thread_slice_id + m * blockDim.x, in_depth);
    }

    UNROLL for (int m = 0; m < n_ILP; m++) { sum += inp[m] * i_n; }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (lane_id == 0) {
      atomicAdd(mean_cache, sum);
    }
    __syncthreads();

    mu = *mean_cache;
    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (lane_id == 0) {
      atomicAdd(std_cache, sqSum);
    }
    __syncthreads();
    if (thread_slice_id == 0) {
      *std_cache = rsqrt(*std_cache * i_n + epsilon);
    }
    __syncthreads();
    rstd = *std_cache;

    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth &&
          thread_id[m] < n_inputs) {
        output[thread_id[m]] = (inp[m] - mu) * rstd;
      }
    }
    __syncthreads();
  }
}

// fused small LN kernel
template <typename T>
__global__ void LayerNormSmallGPUKernel(const LayerNormFusedArgs args,
                                        const T* __restrict__ input,
                                        T* __restrict__ output,
                                        const int num_blocks,
                                        const int slice_per_block) {
  const int slice_size = args.slice_size;
  const int in_depth = args.depth;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  const int slice_id = threadIdx.x / slice_size;
  const int thread_slice_id = threadIdx.x % slice_size;

  T mu;
  T rstd;

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);

    const int thread_id =
        (bId * slice_per_block + slice_id) * in_depth + thread_slice_id;
    // const T inp = 0;
    const T inp = get_value<T>(input + thread_id, thread_slice_id, in_depth);

    mu += inp * i_n;

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      mu += __shfl_xor(mu, mask);
    }

    if (thread_slice_id < in_depth) rstd += (inp - mu) * (inp - mu);

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      rstd += __shfl_xor(rstd, mask);
    }

    rstd = rsqrt(rstd * i_n + epsilon);

    if (thread_slice_id < in_depth && thread_id < n_inputs)
      output[thread_id] = (inp - mu) * rstd;
  }
}

template <typename T, int n_ILP>
__global__ void LayerNormBiasAddGPUKernel(const LayerNormFusedArgs args,
                                          const T* __restrict__ input,
                                          const T* __restrict__ beta,
                                          T* __restrict__ output,
                                          const int num_blocks) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int lane_id = threadIdx.x % warpSize;

  const int thread_slice_id = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* mean_cache = (T*)my_smem;
  T* std_cache = (T*)&my_smem[sizeof(T)];

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);
  T inp[n_ILP];
  T tmp_beta[n_ILP];
  int thread_id[n_ILP];

  T sum;
  T sqSum;
  T mu;
  T rstd;

  UNROLL for (int m = 0; m < n_ILP; m++) {
    tmp_beta[m] = get_value<T>(beta + thread_slice_id + m * blockDim.x,
                               thread_slice_id + m * blockDim.x, in_depth);
  }

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = static_cast<T>(0.0f);
    sqSum = static_cast<T>(0.0f);

    if (thread_slice_id == 0) {
      *mean_cache = static_cast<T>(0.0f);
      *std_cache = static_cast<T>(0.0f);
    }
    __syncthreads();

    UNROLL for (int m = 0; m < n_ILP; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + thread_slice_id;
    }

    UNROLL for (int m = 0; m < n_ILP; m++) {
      inp[m] = get_value<T>(input + thread_id[m],
                            thread_slice_id + m * blockDim.x, in_depth);
    }

    UNROLL for (int m = 0; m < n_ILP; m++) { sum += inp[m] * i_n; }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (lane_id == 0) {
      atomicAdd(mean_cache, sum);
    }
    __syncthreads();

    mu = *mean_cache;
    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (lane_id == 0) {
      atomicAdd(std_cache, sqSum);
    }
    __syncthreads();
    if (thread_slice_id == 0) {
      *std_cache = rsqrt(*std_cache * i_n + epsilon);
    }
    __syncthreads();
    rstd = *std_cache;

    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth &&
          thread_id[m] < n_inputs) {
        output[thread_id[m]] = (inp[m] - mu) * rstd + tmp_beta[m];
      }
    }
    __syncthreads();
  }
}

// fused small LN kernel
template <typename T>
__global__ void LayerNormBiasAddSmallGPUKernel(const LayerNormFusedArgs args,
                                               const T* __restrict__ input,
                                               const T* __restrict__ beta,
                                               T* __restrict__ output,
                                               const int num_blocks,
                                               const int slice_per_block) {
  const int slice_size = args.slice_size;
  const int in_depth = args.depth;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  const int slice_id = threadIdx.x / slice_size;
  const int thread_slice_id = threadIdx.x % slice_size;

  const T tmp_beta =
      get_value<T>(beta + thread_slice_id, thread_slice_id, in_depth);

  T mu;
  T rstd;

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);

    const int thread_id =
        (bId * slice_per_block + slice_id) * in_depth + thread_slice_id;
    // const T inp = 0;
    const T inp = get_value<T>(input + thread_id, thread_slice_id, in_depth);

    mu += inp * i_n;

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      mu += __shfl_xor(mu, mask);
    }

    if (thread_slice_id < in_depth) rstd += (inp - mu) * (inp - mu);

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      rstd += __shfl_xor(rstd, mask);
    }

    rstd = rsqrt(rstd * i_n + epsilon);

    if (thread_slice_id < in_depth && thread_id < n_inputs)
      output[thread_id] = (inp - mu) * rstd + tmp_beta;
  }
}

template <typename T, int n_ILP>
__global__ void LayerNormFusedGPUKernel(const LayerNormFusedArgs args,
                                        const T* __restrict__ input,
                                        const T* __restrict__ gamma,
                                        const T* __restrict__ beta,
                                        T* __restrict__ output,
                                        const int num_blocks) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int lane_id = threadIdx.x % warpSize;

  const int thread_slice_id = threadIdx.x % slice_size;

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* mean_cache = (T*)my_smem;
  T* std_cache = (T*)&my_smem[sizeof(T)];

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);
  T inp[n_ILP];
  T tmp_gamma[n_ILP];
  T tmp_beta[n_ILP];
  int thread_id[n_ILP];

  T sum;
  T sqSum;
  T mu;
  T rstd;

  UNROLL for (int m = 0; m < n_ILP; m++) {
    tmp_gamma[m] = get_value<T>(gamma + thread_slice_id + m * blockDim.x,
                                thread_slice_id + m * blockDim.x, in_depth);
    tmp_beta[m] = get_value<T>(beta + thread_slice_id + m * blockDim.x,
                               thread_slice_id + m * blockDim.x, in_depth);
  }

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    sum = static_cast<T>(0.0f);
    sqSum = static_cast<T>(0.0f);

    if (thread_slice_id == 0) {
      *mean_cache = static_cast<T>(0.0f);
      *std_cache = static_cast<T>(0.0f);
    }
    __syncthreads();

    UNROLL for (int m = 0; m < n_ILP; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + thread_slice_id;
    }

    UNROLL for (int m = 0; m < n_ILP; m++) {
      inp[m] = get_value<T>(input + thread_id[m],
                            thread_slice_id + m * blockDim.x, in_depth);
    }

    UNROLL for (int m = 0; m < n_ILP; m++) { sum += inp[m] * i_n; }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sum += __shfl_xor(sum, mask);
    }
    if (lane_id == 0) {
      atomicAdd(mean_cache, sum);
    }
    __syncthreads();

    mu = *mean_cache;
    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth)
        sqSum += (inp[m] - mu) * (inp[m] - mu);
    }
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
      sqSum += __shfl_xor(sqSum, mask);
    }
    if (lane_id == 0) {
      atomicAdd(std_cache, sqSum);
    }
    __syncthreads();
    if (thread_slice_id == 0) {
      *std_cache = rsqrt(*std_cache * i_n + epsilon);
    }
    __syncthreads();
    rstd = *std_cache;

    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth &&
          thread_id[m] < n_inputs) {
        output[thread_id[m]] =
            (inp[m] - mu) * rstd * tmp_gamma[m] + tmp_beta[m];
      }
    }
    __syncthreads();
  }
}

// fused small LN kernel
template <typename T>
__global__ void LayerNormFusedSmallGPUKernel(
    const LayerNormFusedArgs args, const T* __restrict__ input,
    const T* __restrict__ gamma, const T* __restrict__ beta,
    T* __restrict__ output, const int num_blocks, const int slice_per_block) {
  const int slice_size = args.slice_size;
  const int in_depth = args.depth;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  const int slice_id = threadIdx.x / slice_size;
  const int thread_slice_id = threadIdx.x % slice_size;

  const T tmp_gamma =
      get_value<T>(gamma + thread_slice_id, thread_slice_id, in_depth);
  const T tmp_beta =
      get_value<T>(beta + thread_slice_id, thread_slice_id, in_depth);

  T mu;
  T rstd;

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);

    const int thread_id =
        (bId * slice_per_block + slice_id) * in_depth + thread_slice_id;
    const T inp = get_value<T>(input + thread_id, thread_slice_id, in_depth);

    mu += inp * i_n;

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      mu += __shfl_xor(mu, mask);
    }

    if (thread_slice_id < in_depth) rstd += (inp - mu) * (inp - mu);

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      rstd += __shfl_xor(rstd, mask);
    }

    rstd = rsqrt(rstd * i_n + epsilon);

    if (thread_slice_id < in_depth && thread_id < n_inputs)
      output[thread_id] = (inp - mu) * rstd * tmp_gamma + tmp_beta;
  }
}
}  // namespace
#define LN_GPU_KERNEL(n_ILP)                                       \
  LayerNormGPUKernel<T, n_ILP><<<grid_size, block_size, sbytes>>>( \
      args, input, output, num_blocks)
// A simple launch pad to launch the Cuda kernel for Layer Normalization.
template <typename T>
struct LayerNormGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, T* output) {
    const int warp_size = 32;
    if (args.slice_size <= warp_size) {
      const int block_size = 256;
      const int slice_per_block = block_size / args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices, slice_per_block);
      const int grid_size = std::min(120, num_blocks);
      LayerNormSmallGPUKernel<T><<<grid_size, block_size, 0>>>(
          args, input, output, num_blocks, slice_per_block);
    } else {
      // limit the numebr of threads per block to reduce performance hit on
      // __syncthreads.
      int block_size;
      int n_ILP;
      get_block_size(args.slice_size, block_size, n_ILP);
      const int num_blocks = args.n_slices;
      const int sbytes = 2 * sizeof(T);
      const int grid_size = std::min(MAX_GRID_SIZE, num_blocks);
      switch (n_ILP) {
        case 1:
          LN_GPU_KERNEL(1);
          break;
        case 2:
          LN_GPU_KERNEL(2);
          break;
        case 3:
          LN_GPU_KERNEL(3);
          break;
        case 4:
          LN_GPU_KERNEL(4);
          break;
        case 5:
          LN_GPU_KERNEL(5);
          break;
      }
    }
  }
};

template struct LayerNormGPULaunch<float>;
template struct LayerNormGPULaunch<double>;

template <typename T, int n_ILP>
__global__ void LayerNormBackpropGPUKernel(const LayerNormFusedArgs args,
                                           const T* __restrict__ input,
                                           const T* __restrict__ out_back,
                                           T* __restrict__ in_back,
                                           const int num_blocks) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int thread_slice_id = threadIdx.x % slice_size;
  const int lane_id = threadIdx.x % warpSize;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* mean_cache = (T*)my_smem;
  T* std_cache = mean_cache + 1;
  T* dmu_cache = mean_cache + 2;
  T* dstd_cache = mean_cache + 3;

  T inp[n_ILP];
  T dout[n_ILP];

  int thread_id[n_ILP];

  T mu;
  T rstd;
  T dstd;
  T dmu;

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);
    dmu = static_cast<T>(0.0f);
    dstd = static_cast<T>(0.0f);

    if (thread_slice_id == 0) {
      *mean_cache = static_cast<T>(0.0f);
      *std_cache = static_cast<T>(0.0f);
      *dmu_cache = static_cast<T>(0.0f);
      *dstd_cache = static_cast<T>(0.0f);
    }
    __syncthreads();
    UNROLL for (int m = 0; m < n_ILP; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + thread_slice_id;
    }
    UNROLL for (int m = 0; m < n_ILP; m++) {
      inp[m] = get_value<T>(input + thread_id[m],
                            thread_slice_id + m * blockDim.x, in_depth);
      dout[m] = get_value<T>(out_back + thread_id[m],
                             thread_slice_id + m * blockDim.x, in_depth);
    }

    UNROLL for (int m = 0; m < n_ILP; m++) {
      mu += inp[m] * i_n;
      dmu += dout[m] * i_n;
    }

    warpSum<T>(mu, dmu);
    if (lane_id == 0) {
      atomicAdd(mean_cache, mu);
      atomicAdd(dmu_cache, dmu);
    }
    __syncthreads();

    mu = *mean_cache;
    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m];
      }
    }

    warpSum<T>(rstd, dstd);

    if (lane_id == 0) {
      atomicAdd(std_cache, rstd);
      atomicAdd(dstd_cache, dstd);
    }
    __syncthreads();
    if (thread_slice_id == 0) {
      rstd = rsqrt(*std_cache * i_n + epsilon);
      *std_cache = rstd;
      *dmu_cache = *dmu_cache * rstd;
      *dstd_cache = *dstd_cache * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = *std_cache;
    dstd = *dstd_cache;
    dmu = *dmu_cache;

    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth &&
          thread_id[m] < n_inputs) {
        in_back[thread_id[m]] = dout[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void LayerNormSmallBackpropGPUKernel(const LayerNormFusedArgs args,
                                                const T* __restrict__ input,
                                                const T* __restrict__ out_back,
                                                T* __restrict__ in_back,
                                                const int num_blocks,
                                                const int slice_per_block) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int slice_id = threadIdx.x / slice_size;
  const int thread_slice_id = threadIdx.x % slice_size;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  T mu;
  T rstd;
  T dstd;
  T dmu;

  // we need a thread block here to ensure initialization is complete
  __syncthreads();
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);
    dmu = static_cast<T>(0.0f);
    dstd = static_cast<T>(0.0f);

    const int thread_id =
        (bId * slice_per_block + slice_id) * in_depth + thread_slice_id;
    const T inp = get_value<T>(input + thread_id, thread_slice_id, in_depth);
    const T dout =
        get_value<T>(out_back + thread_id, thread_slice_id, in_depth);

    mu += inp * i_n;
    dmu += dout * i_n;

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      mu += __shfl_xor(mu, mask);
      dmu += __shfl_xor(dmu, mask);
    }

    if (thread_slice_id < in_depth) {
      rstd += (inp - mu) * (inp - mu);
      dstd += (inp - mu) * dout;
    }

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      rstd += __shfl_xor(rstd, mask);
      dstd += __shfl_xor(dstd, mask);
    }

    rstd = rsqrt(rstd * i_n + epsilon);
    dmu = dmu * rstd;
    dstd = dstd * rstd * rstd * rstd * i_n;

    if (thread_slice_id < in_depth && thread_id < n_inputs)
      in_back[thread_id] = dout * rstd - (inp - mu) * dstd - dmu;
  }
}

#define LN_GPU_BACKPROP_KERNEL(n_ILP)                                      \
  LayerNormBackpropGPUKernel<T, n_ILP><<<grid_size, block_size, sbytes>>>( \
      args, input, out_back, in_back, num_blocks)
template <typename T>
struct LayerNormBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* out_back, T* in_back) {
    const int warp_size = 32;
    if (args.slice_size <= warp_size) {
      const int block_size = 128;
      const int slice_per_block = block_size / args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices, slice_per_block);
      const int grid_size = std::min(120, num_blocks);
      const int sbytes = 0;
      // printf("slice_per_block:%d,grid_size:%d\n",slice_per_block,grid_size);
      LayerNormSmallBackpropGPUKernel<T><<<grid_size, block_size, sbytes>>>(
          args, input, out_back, in_back, num_blocks, slice_per_block);
    } else {
      // limit the numebr of threads per block to reduce performance hit on
      // __syncthreads.
      int block_size;
      int n_ILP;
      get_block_size(args.slice_size, block_size, n_ILP);
      const int num_blocks = args.n_slices;
      const int sbytes = 4 * sizeof(T);
      const int max_grid =
          args.n_slices < 2 * MAX_GRID_SIZE ? 60 : MAX_GRID_SIZE;
      const int grid_size = std::min(max_grid, num_blocks);
      switch (n_ILP) {
        case 1:
          LN_GPU_BACKPROP_KERNEL(1);
          break;
        case 2:
          LN_GPU_BACKPROP_KERNEL(2);
          break;
        case 3:
          LN_GPU_BACKPROP_KERNEL(3);
          break;
        case 4:
          LN_GPU_BACKPROP_KERNEL(4);
          break;
        case 5:
          LN_GPU_BACKPROP_KERNEL(5);
          break;
      }
    }
  }
};

template struct LayerNormBackpropGPULaunch<float>;
template struct LayerNormBackpropGPULaunch<double>;

#define LN_GPU_BIASADD_KERNEL(n_ILP)                                      \
  LayerNormBiasAddGPUKernel<T, n_ILP><<<grid_size, block_size, sbytes>>>( \
      args, input, beta, output, num_blocks)
// A simple launch pad to launch the Cuda kernel for Layer Normalization.
template <typename T>
struct LayerNormBiasAddGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* beta, T* output) {
    const int warp_size = 32;
    if (args.slice_size <= warp_size) {
      const int block_size = 256;
      const int slice_per_block = block_size / args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices, slice_per_block);
      const int grid_size = std::min(120, num_blocks);
      LayerNormBiasAddSmallGPUKernel<T><<<grid_size, block_size, 0>>>(
          args, input, beta, output, num_blocks, slice_per_block);
    } else {
      // limit the numebr of threads per block to reduce performance hit on
      // __syncthreads.
      int block_size;
      int n_ILP;
      get_block_size(args.slice_size, block_size, n_ILP);
      const int num_blocks = args.n_slices;
      const int sbytes = 2 * sizeof(T);
      const int grid_size = std::min(MAX_GRID_SIZE, num_blocks);
      switch (n_ILP) {
        case 1:
          LN_GPU_BIASADD_KERNEL(1);
          break;
        case 2:
          LN_GPU_BIASADD_KERNEL(2);
          break;
        case 3:
          LN_GPU_BIASADD_KERNEL(3);
          break;
        case 4:
          LN_GPU_BIASADD_KERNEL(4);
          break;
        case 5:
          LN_GPU_BIASADD_KERNEL(5);
          break;
      }
    }
  }
};

template struct LayerNormBiasAddGPULaunch<float>;
template struct LayerNormBiasAddGPULaunch<double>;

template <typename T, int n_ILP>
__global__ void LayerNormBiasAddBackpropGPUKernel(
    const LayerNormFusedArgs args, const T* __restrict__ input,
    const T* __restrict__ out_back, T* __restrict__ in_back,
    T* __restrict__ beta_back, const int num_blocks) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int thread_slice_id = threadIdx.x % slice_size;
  const int lane_id = threadIdx.x % warpSize;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* mean_cache = (T*)my_smem;
  T* std_cache = mean_cache + 1;
  T* dmu_cache = mean_cache + 2;
  T* dstd_cache = mean_cache + 3;

  T inp[n_ILP];
  T dout[n_ILP];

  T tmp_beta_bp[n_ILP];
  int thread_id[n_ILP];

  UNROLL for (int m = 0; m < n_ILP; m++) {
    tmp_beta_bp[m] = static_cast<T>(0.0f);
  }

  T mu;
  T rstd;
  T dstd;
  T dmu;

  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);
    dmu = static_cast<T>(0.0f);
    dstd = static_cast<T>(0.0f);

    if (thread_slice_id == 0) {
      *mean_cache = static_cast<T>(0.0f);
      *std_cache = static_cast<T>(0.0f);
      *dmu_cache = static_cast<T>(0.0f);
      *dstd_cache = static_cast<T>(0.0f);
    }
    __syncthreads();
    UNROLL for (int m = 0; m < n_ILP; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + thread_slice_id;
    }
    UNROLL for (int m = 0; m < n_ILP; m++) {
      inp[m] = get_value<T>(input + thread_id[m],
                            thread_slice_id + m * blockDim.x, in_depth);
      dout[m] = get_value<T>(out_back + thread_id[m],
                             thread_slice_id + m * blockDim.x, in_depth);
    }

    UNROLL for (int m = 0; m < n_ILP; m++) {
      tmp_beta_bp[m] += dout[m];
      mu += inp[m] * i_n;
      dmu += dout[m] * i_n;
    }

    warpSum<T>(mu, dmu);
    if (lane_id == 0) {
      atomicAdd(mean_cache, mu);
      atomicAdd(dmu_cache, dmu);
    }
    __syncthreads();

    mu = *mean_cache;
    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m];
      }
    }

    warpSum<T>(rstd, dstd);

    if (lane_id == 0) {
      atomicAdd(std_cache, rstd);
      atomicAdd(dstd_cache, dstd);
    }
    __syncthreads();
    if (thread_slice_id == 0) {
      rstd = rsqrt(*std_cache * i_n + epsilon);
      *std_cache = rstd;
      *dmu_cache = *dmu_cache * rstd;
      *dstd_cache = *dstd_cache * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = *std_cache;
    dstd = *dstd_cache;
    dmu = *dmu_cache;

    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (thread_slice_id + m * blockDim.x < in_depth &&
          thread_id[m] < n_inputs) {
        in_back[thread_id[m]] = dout[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    __syncthreads();
  }
  UNROLL for (int m = 0; m < n_ILP; m++) {
    if (thread_slice_id + m * blockDim.x < in_depth) {
      atomicAdd(beta_back + thread_slice_id + m * blockDim.x, tmp_beta_bp[m]);
    }
  }
}

template <typename T>
__global__ void LayerNormBiasAddSmallBackpropGPUKernel(
    const LayerNormFusedArgs args, const T* __restrict__ input,
    const T* __restrict__ out_back, T* __restrict__ in_back,
    T* __restrict__ beta_back, const int num_blocks,
    const int slice_per_block) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int slice_id = threadIdx.x / slice_size;
  const int thread_slice_id = threadIdx.x % slice_size;
  const int lane_id = threadIdx.x % warpSize;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* beta_cache = (T*)&my_smem[in_depth * sizeof(T)];
  // initialize shared memory cache to 0.0
  if (threadIdx.x < in_depth) {
    beta_cache[threadIdx.x] = static_cast<T>(0.0f);
  }

  T mu;
  T rstd;
  T dstd;
  T dmu;

  T tmp_beta_bp = static_cast<T>(0.0f);
  // we need a thread block here to ensure initialization is complete
  __syncthreads();
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);
    dmu = static_cast<T>(0.0f);
    dstd = static_cast<T>(0.0f);

    const int thread_id =
        (bId * slice_per_block + slice_id) * in_depth + thread_slice_id;
    const T inp = get_value<T>(input + thread_id, thread_slice_id, in_depth);
    const T dout =
        get_value<T>(out_back + thread_id, thread_slice_id, in_depth);

    tmp_beta_bp += dout;
    mu += inp * i_n;
    dmu += dout * i_n;

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      mu += __shfl_xor(mu, mask);
      dmu += __shfl_xor(dmu, mask);
    }

    if (thread_slice_id < in_depth) {
      rstd += (inp - mu) * (inp - mu);
      dstd += (inp - mu) * dout;
    }

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      rstd += __shfl_xor(rstd, mask);
      dstd += __shfl_xor(dstd, mask);
    }

    rstd = rsqrt(rstd * i_n + epsilon);
    dmu = dmu * rstd;
    dstd = dstd * rstd * rstd * rstd * i_n;

    if (thread_slice_id < in_depth && thread_id < n_inputs)
      in_back[thread_id] = dout * rstd - (inp - mu) * dstd - dmu;
  }
  for (int mask = slice_size; mask < warpSize; mask *= 2) {
    tmp_beta_bp += __shfl_xor(tmp_beta_bp, mask);
  }
  // accumulate *_bp into shared memory.
  if (lane_id < in_depth) {
    atomicAdd(beta_cache + thread_slice_id, tmp_beta_bp);
  }
  // add *_bp into global memory.
  __syncthreads();
  if (slice_id == 0 && thread_slice_id < in_depth) {
    atomicAdd(beta_back + thread_slice_id, beta_cache[thread_slice_id]);
  }
}

template <typename T>
__global__ void initialize_beta_with_zeros(T* beta_back, const int n_inputs) {
  const int thread_id = threadIdx.x + blockDim.x * blockIdx.x;
  if (thread_id < n_inputs) {
    beta_back[thread_id] = static_cast<T>(0.0f);
  }
}

template <typename T>
void initialize_beta(const LayerNormFusedArgs args, T* beta_back) {
  const int block_size = std::min(args.depth, 256);
  const int grid_size = get_num_blocks(args.depth, block_size);
  initialize_beta_with_zeros<T><<<grid_size, block_size>>>(beta_back,
                                                           args.depth);
}
#define LN_GPU_BIASADD_BACKPROP_KERNEL(n_ILP)                                  \
  LayerNormBiasAddBackpropGPUKernel<T,                                         \
                                    n_ILP><<<grid_size, block_size, sbytes>>>( \
      args, input, out_back, in_back, beta_back, num_blocks)
template <typename T>
struct LayerNormBiasAddBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* out_back, T* in_back, T* beta_back) {
    const int warp_size = 32;
    initialize_beta<T>(args, beta_back);
    if (args.slice_size <= warp_size) {
      const int block_size = 128;
      const int slice_per_block = block_size / args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices, slice_per_block);
      const int grid_size = std::min(120, num_blocks);
      const int sbytes = (2 * args.depth) * sizeof(T);
      // printf("slice_per_block:%d,grid_size:%d\n",slice_per_block,grid_size);
      LayerNormBiasAddSmallBackpropGPUKernel<
          T><<<grid_size, block_size, sbytes>>>(args, input, out_back, in_back,
                                                beta_back, num_blocks,
                                                slice_per_block);
    } else {
      // limit the numebr of threads per block to reduce performance hit on
      // __syncthreads.
      int block_size;
      int n_ILP;
      get_block_size(args.slice_size, block_size, n_ILP);
      const int num_blocks = args.n_slices;
      const int sbytes = 4 * sizeof(T);
      const int max_grid =
          args.n_slices < 2 * MAX_GRID_SIZE ? 60 : MAX_GRID_SIZE;
      const int grid_size = std::min(max_grid, num_blocks);
      switch (n_ILP) {
        case 1:
          LN_GPU_BIASADD_BACKPROP_KERNEL(1);
          break;
        case 2:
          LN_GPU_BIASADD_BACKPROP_KERNEL(2);
          break;
        case 3:
          LN_GPU_BIASADD_BACKPROP_KERNEL(3);
          break;
        case 4:
          LN_GPU_BIASADD_BACKPROP_KERNEL(4);
          break;
        case 5:
          LN_GPU_BIASADD_BACKPROP_KERNEL(5);
          break;
      }
    }
  }
};

template struct LayerNormBiasAddBackpropGPULaunch<float>;
template struct LayerNormBiasAddBackpropGPULaunch<double>;

#define LN_GPU_FUSED_KERNEL(n_ILP)                                      \
  LayerNormFusedGPUKernel<T, n_ILP><<<grid_size, block_size, sbytes>>>( \
      args, input, gamma, beta, output, num_blocks)
// A simple launch pad to launch the Cuda kernel for Layer Normalization.
template <typename T>
struct LayerNormFusedGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* gamma, const T* beta, T* output) {
    const int warp_size = 32;

    if (args.slice_size <= warp_size) {
      const int block_size = 256;
      const int slice_per_block = block_size / args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices, slice_per_block);
      const int grid_size = std::min(120, num_blocks);
      // printf("slice_per_block:%d,grid_size:%d\n",slice_per_block,grid_size);
      LayerNormFusedSmallGPUKernel<T><<<grid_size, block_size, 0>>>(
          args, input, gamma, beta, output, num_blocks, slice_per_block);
    } else {
      // limit the numebr of threads per block to reduce performance hit on
      // __syncthreads.
      int block_size;
      int n_ILP;
      get_block_size(args.slice_size, block_size, n_ILP);
      const int num_blocks = args.n_slices;
      const int sbytes = 2 * sizeof(T);
      // printf("n_ILP:%d,bs:%d,nb:%d,spb:%d\n",n_ILP,block_size,num_blocks,slice_per_block);
      const int grid_size = std::min(MAX_GRID_SIZE, num_blocks);
      switch (n_ILP) {
        case 1:
          LN_GPU_FUSED_KERNEL(1);
          break;
        case 2:
          LN_GPU_FUSED_KERNEL(2);
          break;
        case 3:
          LN_GPU_FUSED_KERNEL(3);
          break;
        case 4:
          LN_GPU_FUSED_KERNEL(4);
          break;
        case 5:
          LN_GPU_FUSED_KERNEL(5);
          break;
      }
    }
  }
};

template struct LayerNormFusedGPULaunch<float>;
template struct LayerNormFusedGPULaunch<double>;

template <typename T, int n_ILP>
__global__ void LayerNormFusedBackpropGPUKernel(
    const LayerNormFusedArgs args, const T* __restrict__ input,
    const T* __restrict__ out_back, const T* __restrict__ gamma,
    T* __restrict__ in_back, T* __restrict__ gamma_back,
    T* __restrict__ beta_back, const int num_blocks) {
  const int in_depth = args.depth;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int lane_id = threadIdx.x % warpSize;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* mean_cache = (T*)my_smem;
  T* std_cache = mean_cache + 1;
  T* dmu_cache = mean_cache + 2;
  T* dstd_cache = mean_cache + 3;

  T inp[n_ILP];
  T dout[n_ILP];

  T tmp_gamma[n_ILP];
  T tmp_gamma_bp[n_ILP];
  T tmp_beta_bp[n_ILP];
  int thread_id[n_ILP];

  UNROLL for (int m = 0; m < n_ILP; m++) {
    tmp_beta_bp[m] = static_cast<T>(0.0f);
    tmp_gamma_bp[m] = static_cast<T>(0.0f);
    tmp_gamma[m] = get_value<T>(gamma + threadIdx.x + m * blockDim.x,
                                threadIdx.x + m * blockDim.x, in_depth);
  }

  T mu;
  T rstd;
  T dstd;
  T dmu;
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);
    dmu = static_cast<T>(0.0f);
    dstd = static_cast<T>(0.0f);

    if (threadIdx.x == 0) {
      *mean_cache = static_cast<T>(0.0f);
      *std_cache = static_cast<T>(0.0f);
      *dmu_cache = static_cast<T>(0.0f);
      *dstd_cache = static_cast<T>(0.0f);
    }
    __syncthreads();
    UNROLL for (int m = 0; m < n_ILP; m++) {
      thread_id[m] = bId * in_depth + m * blockDim.x + threadIdx.x;
    }
    UNROLL for (int m = 0; m < n_ILP; m++) {
      inp[m] = get_value<T>(input + thread_id[m], threadIdx.x + m * blockDim.x,
                            in_depth);
      dout[m] = get_value<T>(out_back + thread_id[m],
                             threadIdx.x + m * blockDim.x, in_depth);
    }

    UNROLL for (int m = 0; m < n_ILP; m++) {
      tmp_beta_bp[m] += dout[m];
      mu += inp[m] * i_n;
      dmu += dout[m] * tmp_gamma[m] * i_n;
    }

    warpSum<T>(mu, dmu);
    if (lane_id == 0) {
      atomicAdd(mean_cache, mu);
      atomicAdd(dmu_cache, dmu);
    }
    __syncthreads();

    mu = *mean_cache;
    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth) {
        rstd += (inp[m] - mu) * (inp[m] - mu);
        dstd += (inp[m] - mu) * dout[m] * tmp_gamma[m];
      }
    }

    warpSum<T>(rstd, dstd);

    if (lane_id == 0) {
      atomicAdd(std_cache, rstd);
      atomicAdd(dstd_cache, dstd);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      rstd = rsqrt(*std_cache * i_n + epsilon);
      *std_cache = rstd;
      *dmu_cache = *dmu_cache * rstd;
      *dstd_cache = *dstd_cache * rstd * rstd * rstd * i_n;
    }
    __syncthreads();
    rstd = *std_cache;
    dstd = *dstd_cache;
    dmu = *dmu_cache;

    UNROLL for (int m = 0; m < n_ILP; m++) {
      if (threadIdx.x + m * blockDim.x < in_depth && thread_id[m] < n_inputs) {
        tmp_gamma_bp[m] += dout[m] * (inp[m] - mu) * rstd;
        in_back[thread_id[m]] =
            dout[m] * tmp_gamma[m] * rstd - (inp[m] - mu) * dstd - dmu;
      }
    }
    __syncthreads();
  }

  UNROLL for (int m = 0; m < n_ILP; m++) {
    if (threadIdx.x + m * blockDim.x < in_depth) {
      atomicAdd(gamma_back + threadIdx.x + m * blockDim.x, tmp_gamma_bp[m]);
      atomicAdd(beta_back + threadIdx.x + m * blockDim.x, tmp_beta_bp[m]);
    }
  }
}

template <typename T>
__global__ void LayerNormFusedSmallBackpropGPUKernel(
    const LayerNormFusedArgs args, const T* __restrict__ input,
    const T* __restrict__ out_back, const T* __restrict__ gamma,
    T* __restrict__ in_back, T* __restrict__ gamma_back,
    T* __restrict__ beta_back, const int num_blocks,
    const int slice_per_block) {
  const int in_depth = args.depth;
  const int slice_size = args.slice_size;
  const int n_inputs = args.n_inputs;
  const T epsilon = args.epsilon;

  const int slice_id = threadIdx.x / slice_size;
  const int thread_slice_id = threadIdx.x % slice_size;
  const int lane_id = threadIdx.x % warpSize;

  const T i_n = static_cast<T>(1.0f) / static_cast<T>(in_depth);

  extern __shared__ __align__(sizeof(T)) unsigned char my_smem[];
  T* gamma_cache = (T*)my_smem;
  T* beta_cache = (T*)&my_smem[in_depth * sizeof(T)];
  // initialize shared memory cache to 0.0
  if (threadIdx.x < in_depth) {
    gamma_cache[threadIdx.x] = static_cast<T>(0.0f);
    beta_cache[threadIdx.x] = static_cast<T>(0.0f);
  }

  const T tmp_gamma =
      get_value<T>(gamma + thread_slice_id, thread_slice_id, in_depth);
  T mu;
  T rstd;
  T dstd;
  T dmu;

  T tmp_gamma_bp = static_cast<T>(0.0f);
  T tmp_beta_bp = static_cast<T>(0.0f);
  // we need a thread block here to ensure initialization is complete
  __syncthreads();
  for (int bId = blockIdx.x; bId < num_blocks; bId += gridDim.x) {
    mu = static_cast<T>(0.0f);
    rstd = static_cast<T>(0.0f);
    dmu = static_cast<T>(0.0f);
    dstd = static_cast<T>(0.0f);

    const int thread_id =
        (bId * slice_per_block + slice_id) * in_depth + thread_slice_id;
    const T inp = get_value<T>(input + thread_id, thread_slice_id, in_depth);
    const T dout =
        get_value<T>(out_back + thread_id, thread_slice_id, in_depth);

    const T dout_g = dout * tmp_gamma;
    tmp_beta_bp += dout;
    mu += inp * i_n;
    dmu += dout * tmp_gamma * i_n;

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      mu += __shfl_xor(mu, mask);
      dmu += __shfl_xor(dmu, mask);
    }

    if (thread_slice_id < in_depth) {
      rstd += (inp - mu) * (inp - mu);
      dstd += (inp - mu) * dout_g;
    }

    for (int mask = slice_size / 2; mask > 0; mask /= 2) {
      rstd += __shfl_xor(rstd, mask);
      dstd += __shfl_xor(dstd, mask);
    }

    rstd = rsqrt(rstd * i_n + epsilon);
    dmu = dmu * rstd;
    dstd = dstd * rstd * rstd * rstd * i_n;

    if (thread_slice_id < in_depth && thread_id < n_inputs) {
      tmp_gamma_bp += dout * (inp - mu) * rstd;
      in_back[thread_id] = dout_g * rstd - (inp - mu) * dstd - dmu;
    }
  }
  for (int mask = slice_size; mask < warpSize; mask *= 2) {
    tmp_gamma_bp += __shfl_xor(tmp_gamma_bp, mask);
    tmp_beta_bp += __shfl_xor(tmp_beta_bp, mask);
  }
  // accumulate *_bp into shared memory.
  if (lane_id < in_depth) {
    atomicAdd(gamma_cache + thread_slice_id, tmp_gamma_bp);
    atomicAdd(beta_cache + thread_slice_id, tmp_beta_bp);
  }
  // add *_bp into global memory.
  __syncthreads();
  if (slice_id == 0 && thread_slice_id < in_depth) {
    atomicAdd(gamma_back + thread_slice_id, gamma_cache[thread_slice_id]);
    atomicAdd(beta_back + thread_slice_id, beta_cache[thread_slice_id]);
  }
}

template <typename T>
__global__ void initialize_with_zeros(T* gamma_back, T* beta_back,
                                      const int n_inputs) {
  const int thread_id = threadIdx.x + blockDim.x * (blockIdx.x / 2);
  if (thread_id < n_inputs) {
    if (blockIdx.x % 2 == 0) {
      gamma_back[thread_id] = static_cast<T>(0.0f);
    } else
      beta_back[thread_id] = static_cast<T>(0.0f);
  }
}

template <typename T>
void initialize_outputs(const LayerNormFusedArgs args, T* gamma_back,
                        T* beta_back) {
  const int block_size = std::min(args.depth, 256);
  const int grid_size = get_num_blocks(args.depth, block_size) * 2;
  initialize_with_zeros<T><<<grid_size, block_size>>>(gamma_back, beta_back,
                                                      args.depth);
}
#define LN_GPU_FUSED_BACKPROP_KERNEL(n_ILP)                                  \
  LayerNormFusedBackpropGPUKernel<T,                                         \
                                  n_ILP><<<grid_size, block_size, sbytes>>>( \
      args, input, out_back, gamma, in_back, gamma_back, beta_back,          \
      num_blocks)
template <typename T>
struct LayerNormFusedBackpropGPULaunch {
  static void Run(const GPUDevice& d, const LayerNormFusedArgs args,
                  const T* input, const T* out_back, const T* gamma, T* in_back,
                  T* gamma_back, T* beta_back) {
    const int warp_size = 32;
    initialize_outputs<T>(args, gamma_back, beta_back);
    if (args.slice_size <= warp_size) {
      const int block_size = 128;
      const int slice_per_block = block_size / args.slice_size;
      const int num_blocks = get_num_blocks(args.n_slices, slice_per_block);
      const int grid_size = std::min(120, num_blocks);
      const int sbytes = (2 * args.depth) * sizeof(T);
      LayerNormFusedSmallBackpropGPUKernel<
          T><<<grid_size, block_size, sbytes>>>(args, input, out_back, gamma,
                                                in_back, gamma_back, beta_back,
                                                num_blocks, slice_per_block);
    } else {
      // limit the numebr of threads per block to reduce performance hit on
      // __syncthreads.
      int block_size;
      int n_ILP;
      get_block_size(args.slice_size, block_size, n_ILP);
      const int num_blocks = args.n_slices;
      const int sbytes = 4 * sizeof(T);
      const int max_grid =
          args.n_slices < 2 * MAX_GRID_SIZE ? 60 : MAX_GRID_SIZE;
      const int grid_size = std::min(max_grid, num_blocks);
      switch (n_ILP) {
        case 1:
          LN_GPU_FUSED_BACKPROP_KERNEL(1);
          break;
        case 2:
          LN_GPU_FUSED_BACKPROP_KERNEL(2);
          break;
        case 3:
          LN_GPU_FUSED_BACKPROP_KERNEL(3);
          break;
        case 4:
          LN_GPU_FUSED_BACKPROP_KERNEL(4);
          break;
        case 5:
          LN_GPU_FUSED_BACKPROP_KERNEL(5);
          break;
      }
    }
  }
};

template struct LayerNormFusedBackpropGPULaunch<float>;
template struct LayerNormFusedBackpropGPULaunch<double>;
}  // namespace tensorflow
#endif  // GOOGLE_CUDA
