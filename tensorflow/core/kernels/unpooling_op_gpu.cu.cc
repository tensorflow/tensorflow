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
__global__ void UnpoolForwardWithIndexKernel(const int numThreads, const T* pooledData, const tensorflow::int64* indices, T* unpooledData)
{
  CUDA_1D_KERNEL_LOOP(index, numThreads)
  {
    int64 unpooledIndex = indices[index];
    unpooledData[unpooledIndex] = pooledData[index];
  }
}

bool UnpoolForwardWithIndex(const float* input, TensorShape inputShape, const int64* indices, float* unpooledData, const Eigen::GpuDevice& device)
{
  const int threadsPerBlock = 1024;

  const int64 numPooledPoints = GetTensorDim(inputShape, FORMAT_NHWC, 'N')*GetTensorDim(inputShape, FORMAT_NHWC, 'H')*GetTensorDim(inputShape, FORMAT_NHWC, 'W')*GetTensorDim(inputShape, FORMAT_NHWC, 'C');

  tensorflow::int64 numBlocks = (numPooledPoints+threadsPerBlock-1)/threadsPerBlock;
  UnpoolForwardWithIndexKernel<<<numBlocks, threadsPerBlock, 0, device.stream()>>>(numPooledPoints, input, indices, unpooledData);

  return device.ok();
}

}

#endif
