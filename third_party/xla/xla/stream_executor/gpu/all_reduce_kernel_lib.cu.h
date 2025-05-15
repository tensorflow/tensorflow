/* Copyright 2025 The OpenXLA Authors.

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
#ifndef XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_

#include <array>
#include <cstdint>

#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"

namespace stream_executor::gpu {

constexpr int64_t kNumElementsPerThread = 4;

template <typename T>
union Vec;

template <>
union alignas(16) Vec<float> {
  using PackedType = int4;

  float data[4];
  PackedType packed;
};

template <>
union alignas(8) Vec<__nv_bfloat16> {
  using PackedType = int2;

  __nv_bfloat16 data[4];
  PackedType packed;
};

template <typename T>
__device__ __forceinline__ Vec<T> VecLoad(T* addr) {
  Vec<T> vec;
  vec.packed = *(reinterpret_cast<typename Vec<T>::PackedType*>(addr));
  return vec;
}

template <typename T>
__device__ __forceinline__ void VecStore(T* addr, const Vec<T>& vec) {
  *(reinterpret_cast<typename Vec<T>::PackedType*>(addr)) = vec.packed;
}

template <typename T>
__device__ __forceinline__ void VecAdd(Vec<T>& res, const Vec<T>& vec) {
  res.data[0] += vec.data[0];
  res.data[1] += vec.data[1];
  res.data[2] += vec.data[2];
  res.data[3] += vec.data[3];
}

template <typename T>
__global__ void AllReduceKernelImpl(
    std::array<T* __restrict__, kMaxNumAllReduceInputPtrs> input_ptrs,
    T* __restrict__ output_ptr, int64_t num_inputs, int64_t num_elements) {
  int64_t offset =
      kNumElementsPerThread * (blockIdx.x * blockDim.x + threadIdx.x);
  int64_t stride = kNumElementsPerThread * blockDim.x * gridDim.x;

  for (int i = offset; i < num_elements; i += stride) {
    Vec<T> sum;
    sum.data[0] = 0;
    sum.data[1] = 0;
    sum.data[2] = 0;
    sum.data[3] = 0;

#pragma unroll
    for (int j = 0; j < kMaxNumAllReduceInputPtrs; ++j) {
      if (j >= num_inputs) break;

      Vec<T> input_vec = VecLoad(input_ptrs[j] + i);
      VecAdd(sum, input_vec);
    }

    VecStore(output_ptr + i, sum);
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_ALL_REDUCE_KERNEL_LIB_CU_H_
