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

#ifndef XLA_STREAM_EXECUTOR_GPU_REPEAT_BUFFER_KERNEL_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_REPEAT_BUFFER_KERNEL_CU_H_

#include <cstdint>

namespace stream_executor::gpu {

// Populate the last `buffer_size - repeat_size` bytes of `buffer` by repeating
// the first `repeat_size` bytes. This should be launched with at least
// `repeat_size` threads in total.
__global__ void RepeatBufferKernelImpl(char* buffer, int64_t repeat_size,
                                   int64_t buffer_size) {
  int64_t global_index = blockDim.x * blockIdx.x + threadIdx.x;
  if (global_index >= repeat_size) {
    return;
  }
  const char src_value = buffer[global_index];
  for (int64_t dst_index = global_index + repeat_size; dst_index < buffer_size;
       dst_index += repeat_size) {
    buffer[dst_index] = src_value;
  }
}

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_REPEAT_BUFFER_KERNEL_CU_H_
