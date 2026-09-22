/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/integer_pow_gpu.h"

#include <algorithm>
#include <cstdint>

#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace {

template <typename T>
__global__ void CheckNegativeExponentKernel(const T* exponents, int64_t size,
                                            int32_t* found_negative) {
  for (int64_t i : GpuGridRangeX<int64_t>(size)) {
    if (exponents[i] < 0) {
      atomicExch(found_negative, 1);
    }
  }
}

}  // namespace

template <typename T>
absl::Status CheckNegativeExponentGpu<T>::operator()(
    const Eigen::GpuDevice& device, const T* exponents, int64_t size,
    int32_t* found_negative) {
  constexpr int kThreads = 256;
  const int blocks = std::min<int64_t>(1 + (size - 1) / kThreads, 1024);
  return GpuLaunchKernel(CheckNegativeExponentKernel<T>, blocks, kThreads, 0,
                         device.stream(), exponents, size, found_negative);
}

template struct CheckNegativeExponentGpu<int8_t>;
template struct CheckNegativeExponentGpu<int16_t>;
template struct CheckNegativeExponentGpu<int64_t>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
