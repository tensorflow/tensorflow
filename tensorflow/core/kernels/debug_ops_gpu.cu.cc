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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU

#include <assert.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

namespace tensorflow {

namespace {

typedef Eigen::GpuDevice GPUDevice;

// A CUDA kernel that fills a length-2 vector according to whether any of the
// input data contains infinity or NaN. The first element is filled with
// infinity of any of the elements is +/- infinity. The second element is
// filled with NaN if any of the elements is NaN.
template <typename T>
__global__ void ReduceInfNanTwoSlotsKernel(const T* __restrict__ data, int size,
                                           float output[2]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;

  while (offset < size) {
    if (isinf(data[offset])) {
      if (data[offset] < static_cast<T>(0.f)) {
        output[0] = -std::numeric_limits<double>::infinity();
      } else {
        output[1] = std::numeric_limits<double>::infinity();
      }
    }
    if (isnan(data[offset])) {
      output[2] = std::numeric_limits<double>::quiet_NaN();
    }
    offset += total_thread_count;
  }
}

}  // namespace

template <typename T>
struct ReduceInfNanTwoSlotsLaunch {
  void Run(const GPUDevice& d, const T* data, int size, float output[2]) {
    const int32 block_size = d.maxGpuThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor()) /
        block_size;

    TF_CHECK_OK(GpuLaunchKernel(ReduceInfNanTwoSlotsKernel<T>, num_blocks,
                                block_size, 0, d.stream(), data, size, output));
  }
};

template struct ReduceInfNanTwoSlotsLaunch<Eigen::half>;
template struct ReduceInfNanTwoSlotsLaunch<float>;
template struct ReduceInfNanTwoSlotsLaunch<double>;

}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
