/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/libraries/native_custom_call_thunks/write_value_thunk_folded/write_value_folded_kernel.h"

#include <cstdint>

namespace stream_executor::cuda {

// Writes `val` to each element of `out`. The kernel is launched with one grid
// block per output element (block_dims = num_elements, thread_dims = 1), so the
// global index below covers exactly [0, num_elements).
__global__ void WriteValueFoldedKernel(int32_t* out, int32_t val) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  out[index] = val;
}

WriteValueFoldedKernelFn GetWriteValueFoldedKernel() {
  return &WriteValueFoldedKernel;
}

}  // namespace stream_executor::cuda
