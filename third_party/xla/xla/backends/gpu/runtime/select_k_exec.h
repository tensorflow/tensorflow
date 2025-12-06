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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SELECT_K_EXEC_H_
#define XLA_BACKENDS_GPU_RUNTIME_SELECT_K_EXEC_H_

#include <cstdint>

#include "absl/status/status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Launches a Top-K selection on GPU for a batch of matrices.
//
// Args:
//   device_ordinal: GPU device index to run the operation on.
//   allocator: StreamExecutor memory allocator for device buffers.
//   stream: StreamExecutor stream for the GPU operations.
//   data_in: Device memory containing input matrices (batch × n).
//   data_out: Device memory to store top-k values (batch × k).
//   indices_out: Device memory to store top-k indices (batch × k).
//   batch: Number of rows (matrices) in the batch.
//   n: Number of columns (elements per row) in input matrices.
//   k: Number of top elements to select per row.
//
// Returns:
//   absl::Status indicating success or failure of the operation.
template <typename T>
absl::Status select_k_exec(int device_ordinal,
                           ::stream_executor::DeviceAddressAllocator* allocator,
                           ::stream_executor::Stream* stream,
                           ::stream_executor::DeviceAddressBase data_in,
                           ::stream_executor::DeviceAddressBase data_out,
                           ::stream_executor::DeviceAddressBase indices_out,
                           std::uint32_t batch, std::uint32_t n,
                           std::uint32_t k);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SELECT_K_EXEC_H_
