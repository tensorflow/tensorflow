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

#ifndef XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_KERNEL_LIB_CU_H_
#define XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_KERNEL_LIB_CU_H_

#include <array>
#include <cstdint>

#include "xla/stream_executor/gpu/ragged_all_to_all_kernel.h"

namespace stream_executor::gpu {

// RaggedAllToAll instruction performs a collective AllToAll operation on ragged
// tensors. For the semantics of each operand see the documentation of
// `RaggedAllToAll` HLO instruction.
//
// This kernel benefits from direct memory access on GPUs on single host. The
// kernel itself does not do any collective communication. Runtime is
// responsible to gather pointers for output buffers on different devices and
// synchronize streams before and after the kernel.
//
// There are `N` devices that participate in the exchange and to each device we
// need to send `num_updates_per_rank` updates.
// Updates are laid out in row-major order in `input_offsets`, `send_sizes` and
// `output_offsets` buffers.
//
// For i-th update to j-th device the kernel does the following copy:
//  update_idx = i + j * num_updates_per_replica
//
//  input_offset = input_offsets[update_idx]
//  send_size = send_sizes[update_idx]
//  output_offset = output_offsets[update_idx]
//
//  update_slice = input[input_offset: input_offset + send_size]
//  output_ptrs[j][output_offset : output_offset + send_size] = update_slice
//
// Launch parameters:
//  - Block grid: (N*num_updates_per_rank, num_blocks_per_update, 1)
//  - Thread grid: (num_threads_per_update, 1, 1)
template <typename T>
__global__ void __launch_bounds__(128) RaggedAllToAllKernelImpl(
    const T* __restrict__ input_ptr,
    std::array<void* __restrict__, kMaxNumRaggedAllToAllOutputPtrs> output_ptrs,
    const int64_t* __restrict__ input_offsets_ptr,
    const int64_t* __restrict__ send_sizes_ptr,
    const int64_t* __restrict__ output_offsets_ptr,
    int64_t num_updates_per_replica, int64_t num_row_elements) {
  int64_t update_idx = blockIdx.x;
  int64_t output_idx = update_idx / num_updates_per_replica;

  T* output_ptr = absl::bit_cast<T* __restrict__>(output_ptrs[output_idx]);

  int64_t input_offset = input_offsets_ptr[update_idx];
  int64_t send_size = send_sizes_ptr[update_idx];
  int64_t output_offset = output_offsets_ptr[update_idx];

  int64_t input_offset_start = input_offset * num_row_elements;
  int64_t output_offset_start = output_offset * num_row_elements;

  int64_t update_size = send_size * num_row_elements;

  for (int64_t i = threadIdx.x + blockIdx.y * blockDim.x; i < update_size;
       i += blockDim.x * gridDim.y) {
    output_ptr[output_offset_start + i] = input_ptr[input_offset_start + i];
  }
}
}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_RAGGED_ALL_TO_ALL_KERNEL_LIB_CU_H_
