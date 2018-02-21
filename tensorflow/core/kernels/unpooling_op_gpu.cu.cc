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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

template <typename T>
__global__ void UnpoolForwardKernel(const int num_threads, const T* pooled_data, const int64* indices, T* unpooled_data)
{
  CUDA_1D_KERNEL_LOOP(index, num_threads)
  {
    int64 unpooled_index = indices[index];
    unpooled_data[unpooled_index] = pooled_data[index];
  }
}

template <typename T>
__global__ void MaxUnpoolBackward(const int num_threads, const T* unpooled_gradient, const int64* indices, T* pooled_gradient)
{
  CUDA_1D_KERNEL_LOOP(pooled_index, num_threads)
  {
    int64 unpooled_index = indices[pooled_index];
    CudaAtomicAdd(pooled_gradient+pooled_index, unpooled_gradient[unpooled_index]);
  }
}

template <typename T>
__global__ void SetToZero(const int num_threads, T* input)
{
  CUDA_1D_KERNEL_LOOP(index, num_threads)
  {
    *(input + index) = T(0);
  }
}

bool UnpoolForward(const float* input, TensorShape input_shape, const int64* indices, float* unpooled_data, const Eigen::GpuDevice& device)
{
  const int threads_per_block = 1024;

  const int64 num_pooled_points = GetTensorDim(input_shape, FORMAT_NHWC, 'N')*GetTensorDim(input_shape, FORMAT_NHWC, 'H')*GetTensorDim(input_shape, FORMAT_NHWC, 'W')*GetTensorDim(input_shape, FORMAT_NHWC, 'C');

  int64 num_blocks = (num_pooled_points+threads_per_block-1)/threads_per_block;
  UnpoolForwardKernel<<<num_blocks, threads_per_block, 0, device.stream()>>>(num_pooled_points, input, indices, unpooled_data);

  return device.ok();
}

bool UnpoolBackward(const float* unpooled_gradient, const int64* indices, float* pooled_gradient, const int64 num_pooled_points, const Eigen::GpuDevice& device)
{
 const int threads_per_block = 1024;
 int64 num_blocks = (num_pooled_points+threads_per_block-1)/threads_per_block;

 SetToZero<<<num_blocks, threads_per_block, 0, device.stream()>>>(num_pooled_points, pooled_gradient);

 MaxUnpoolBackward<<<num_blocks, threads_per_block, 0, device.stream()>>>(num_pooled_points, unpooled_gradient, indices, pooled_gradient);

 return device.ok();
}

}

#endif
