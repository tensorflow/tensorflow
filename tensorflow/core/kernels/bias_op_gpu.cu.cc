/* Copyright 2015 Google Inc. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <algorithm>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bias_op.h"
#include "tensorflow/core/kernels/bias_op_gpu.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

// Definition of the GPU implementations declared in bias_op.cc.

template <typename T>
__global__ void BiasNHWCKernel(int32 nthreads, const T* input, const T* bias,
                               T* output, int32 bias_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int32 bias_offset = index % bias_size;
    output[index] = ldg(input + index) + ldg(bias + bias_offset);
  }
}

template <typename T>
__global__ void BiasNCHWKernel(int32 nthreads, const T* input, const T* bias,
                               T* output, int32 bias_size, int32 image_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int32 index2 = index / image_size;
    int32 bias_offset = index2 % bias_size;
    output[index] = ldg(input + index) + ldg(bias + bias_offset);
  }
}

// Add "bias" to "input", broadcasting it on all dimensions but the bias
// dimension.
template <typename T>
void BiasGPU<T>::compute(const GPUDevice& d, const T* input, const T* bias,
                         T* output, int32 batch, int32 height, int32 width,
                         int32 channel, TensorFormat data_format) {
  const int32 bias_size = channel;
  const int32 image_size = height * width;
  const int32 total_count = batch * bias_size * image_size;
  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
  if (data_format == FORMAT_NHWC) {
    BiasNHWCKernel<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input, bias, output, bias_size);
  } else {
    BiasNCHWKernel<
        T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input, bias, output, bias_size,
        image_size);
  }
}

// A naive implementation that is functional on all cases.
template <typename T>
__global__ void BiasGradNHWC_Naive(int32 nthreads, const T* output_backprop,
                                   T* bias_backprop, int32 bias_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int32 bias_offset = index % bias_size;
    CudaAtomicAdd(bias_backprop + bias_offset, ldg(output_backprop + index));
  }
}

// A naive implementation that is functional on all cases.
template <typename T>
__global__ void BiasGradNCHW_Naive(int32 nthreads, const T* output_backprop,
                                   T* bias_backprop, int32 bias_size,
                                   int32 image_size) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    int32 index2 = index / image_size;
    int32 bias_offset = index2 % bias_size;
    CudaAtomicAdd(bias_backprop + bias_offset, ldg(output_backprop + index));
  }
}

extern __shared__ char s_buf[];

template <typename T>
__global__ void BiasGradNHWC_SharedAtomics(int32 nthreads,
                                           const T* output_backprop,
                                           T* bias_backprop, int32 bias_size) {
  T* s_data = reinterpret_cast<T*>(s_buf);
  for (int32 index = threadIdx.x; index < bias_size; index += blockDim.x) {
    s_data[index] = 0;
  }
  __syncthreads();

  for (int32 index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int32 bias_offset = index % bias_size;
    CudaAtomicAdd(s_data + bias_offset, ldg(output_backprop + index));
  }
  __syncthreads();

  for (int32 index = threadIdx.x; index < bias_size; index += blockDim.x) {
    CudaAtomicAdd(bias_backprop + index, s_data[index]);
  }
}

template <typename T>
__global__ void BiasGradNCHW_SharedAtomics(int32 nthreads,
                                           const T* output_backprop,
                                           T* bias_backprop, int32 bias_size,
                                           int32 image_size,
                                           int32 shared_replicas) {
  T* s_data = reinterpret_cast<T*>(s_buf);
  int32 s_data_size = bias_size * shared_replicas;
  for (int32 index = threadIdx.x; index < s_data_size; index += blockDim.x) {
    s_data[index] = 0;
  }
  __syncthreads();

  for (int32 index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int32 index2 = index / image_size;
    int32 bias_slot_index = index2 % bias_size;
    int32 bias_slot_offset = index % shared_replicas;
    int32 bias_offset = bias_slot_index * shared_replicas + bias_slot_offset;
    CudaAtomicAdd(s_data + bias_offset, ldg(output_backprop + index));
  }
  __syncthreads();

  for (int32 index = threadIdx.x; index < s_data_size; index += blockDim.x) {
    int bias_slot_index = index / shared_replicas;
    CudaAtomicAdd(bias_backprop + bias_slot_index, s_data[index]);
  }
}

template <typename T>
void BiasGradGPU<T>::compute(const GPUDevice& d, const T* output_backprop,
                             T* bias_backprop, int32 batch, int32 height,
                             int32 width, int32 channel,
                             TensorFormat data_format) {
  const int32 bias_size = channel;
  const int32 image_size = height * width;
  const int32 total_count = batch * bias_size * image_size;
  CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);

  const int max_shared_memory_size = d.sharedMemPerBlock() / 2;
  int32 shared_memory_size = bias_size * sizeof(T);
  int shared_replicas = 1;
  if (data_format == FORMAT_NCHW) {
    // For NCHW, the reduction in the HW dimensions all go to the same locaiton,
    // which causes a lot of bank conflicts. So having a number of them can
    // improve the performance. But we also want to limit their usage so the
    // warp occupancy does not decrease.
    if (shared_memory_size <= max_shared_memory_size) {
      // We need enough shared memory to avoid bank conflict. But not too much
      // so that it would reduce occupancy.
      static constexpr int kMaxSharedReplicas = 8;
      shared_replicas = std::min(kMaxSharedReplicas,
                                 max_shared_memory_size / shared_memory_size);
      shared_memory_size *= shared_replicas;
    }
  }
  // Check if we have enough shared memory.
  if (shared_memory_size <= max_shared_memory_size) {
    if (data_format == FORMAT_NHWC) {
      BiasGradNHWC_SharedAtomics<
          T><<<config.block_count, config.thread_per_block, shared_memory_size,
               d.stream()>>>(total_count, output_backprop, bias_backprop,
                             bias_size);
    } else {
      BiasGradNCHW_SharedAtomics<
          T><<<config.block_count, config.thread_per_block, shared_memory_size,
               d.stream()>>>(total_count, output_backprop, bias_backprop,
                             bias_size, image_size, shared_replicas);
    }
  } else {
    // Note that even if we don't have enough shared memory to fit the entire
    // output block, it is possible to process one group of elements at a time.
    // But for now, we simply fall back to the naive implementation.
    if (data_format == FORMAT_NHWC) {
      BiasGradNHWC_Naive<
          T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          total_count, output_backprop, bias_backprop, bias_size);
    } else {
      BiasGradNCHW_Naive<
          T><<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          total_count, output_backprop, bias_backprop, bias_size, image_size);
    }
  }
}

#define DEFINE_GPU_SPECS(T)   \
  template struct BiasGPU<T>; \
  template struct BiasGradGPU<T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
