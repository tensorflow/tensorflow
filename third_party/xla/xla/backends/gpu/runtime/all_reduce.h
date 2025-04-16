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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns true if the all-reduce kernel is supported for the given number of
// inputs and element type.
bool IsAllReduceKernelSupported(int64_t num_inputs, PrimitiveType element_type);

// Performs element-wise addition of all input buffers and stores the result in
// the output buffer.
// The kernel is intended to be used for all-reduce operations in environment
// where direct peer memory access is available. Input buffers can point to
// memory on different devices. The caller is responsible to gather pointers
// from different devices.
//
// TODO(b/383125489): Add synchronization between blocks in the kernek.
// The caller is also responsible to synchronize streams on all participating
// devices before and after the kernel execution.
//
// Input arguments:
//  - input_buffers: A list of input buffers.
//  - output_buffer: The buffer to store the result.
//  - num_inputs: The number of input buffers.
//  - num_elements: The number of elements in each buffer.
absl::Status RunAllReduceKernel(
    se::Stream* stream, PrimitiveType element_type,
    absl::Span<const se::DeviceMemoryBase> input_buffers,
    se::DeviceMemoryBase output_buffer, int64_t num_inputs,
    int64_t num_elements);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_H_
