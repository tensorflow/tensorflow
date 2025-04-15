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

#include <array>
#include <cstdint>

#include "xla/service/gpu/kernels/all_reduce_kernel_common.h"

namespace xla::gpu {
namespace {

template <typename T>
__global__ void AllReduceKernel(
    std::array<void* __restrict__, kMaxNumAllReduceInputPtrs> input_ptrs,
    T* __restrict__ output_ptr, int64_t num_inputs, int64_t num_elements) {
  int64_t offset = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = blockDim.x * gridDim.x;

  for (int i = offset; i < num_elements; i += stride) {
    T sum = 0;

#pragma unroll
    for (int j = 0; j < kMaxNumAllReduceInputPtrs; ++j) {
      if (j >= num_inputs) break;

      // TODO(b/383125489): Add vectorization.
      T* input_ptr =
          reinterpret_cast<  // REINTERPRET_CAST_OK=tsl::safe_reinterpret_cast
                             // doesn't work with __restrict__.
              T* __restrict__>(input_ptrs[j]);
      sum += input_ptr[i];
    }

    output_ptr[i] = sum;
  }
}

}  // namespace

template <typename T>
void* GetAllReduceKernel() {
  return reinterpret_cast<  // REINTERPRET_CAST_OK=tsl::safe_reinterpret_cast
                            // doesn't support this cast, but it's necessary to
                            // conform to se::TypedKernelFactory<>::Create().
      void*>(&AllReduceKernel<T>);
}

template void* GetAllReduceKernel<float>();

}  // namespace xla::gpu
