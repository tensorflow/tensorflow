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

#include <assert.h>
#include <stdio.h>

#include <math.h>
#include <algorithm>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_launch_config.h"

namespace tensorflow {

namespace {

typedef Eigen::GpuDevice GPUDevice;

// A Cuda kernel to check if each element is Inf or Nan. If any exists, the
// relevant elements in abnormal_detected will be set
template <typename T>
__global__ void CheckNumericsKernel(const T *data, int size,
                                    int abnormal_detected[2]) {
  const int32 thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const int32 total_thread_count = gridDim.x * blockDim.x;

  int32 offset = thread_id;

  while (offset < size) {
    if (isnan(data[offset])) {
      abnormal_detected[0] = 1;
    }
    if (isinf(data[offset])) {
      abnormal_detected[1] = 1;
    }
    offset += total_thread_count;
  }
}

}  // namespace

// A simple launch pad to launch the Cuda kernels that checks the numerical
// abnormality in the given array
template <typename T>
struct CheckNumericsLaunch {
  void Run(const GPUDevice &d, const T *data, int size,
           int abnormal_detected[2]) {
    const int32 block_size = d.maxGpuThreadsPerBlock();
    const int32 num_blocks =
        (d.getNumGpuMultiProcessors() * d.maxGpuThreadsPerMultiProcessor()) /
        block_size;

    TF_CHECK_OK(CudaLaunchKernel(CheckNumericsKernel<T>, num_blocks, block_size,
                                 0, d.stream(), data, size, abnormal_detected));
  }
};

template struct CheckNumericsLaunch<Eigen::half>;
template struct CheckNumericsLaunch<float>;
template struct CheckNumericsLaunch<double>;

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
