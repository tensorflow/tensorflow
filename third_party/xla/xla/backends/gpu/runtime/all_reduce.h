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
#include "xla/core/collectives/rank_id.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns true if the all-reduce kernel is supported for the given number of
// inputs, elements and element type.
bool IsAllReduceKernelSupported(int64_t num_inputs, int64_t num_elements,
                                PrimitiveType element_type);

// Performs element-wise addition of all input buffers and stores the result in
// the output buffer.
// The kernel is intended to be used for all-reduce operations in environment
// where direct peer memory access is available. Input buffers can point to
// memory on different devices. The caller is responsible to gather pointers
// from different devices.
//
// The kernel copies data from local input buffer to remote input buffer of the
// current rank at the start of the kernel.
//
// The kernel performs synchronization across devices at the start and the end
// of the kernel. The synchronization happens between blocks with the same id.
//
// Input arguments:
//  - remove_input_buffers: A list of buffers with inputs on other devices.
//    The data in the buffers maybe not be initialized until blocks on different
//    devices are synchronized.
//  - local_input_buffer: The buffer with local input. Can be the same as
//    the output buffer.
//  - output_buffer: The buffer to store the result.
//  - rank: Identifier of the device.
//  - num_ranks: The number of devices participating in the operation.
//  - num_elements: The number of elements in each buffer.
//  - signal_flags_buffers: A list of buffers with signal flags that are used to
//    synchronize blocks on different devices. The size of each signal buffer
//    should be equal to the `num_ranks * num_blocks`.
absl::Status RunAllReduceKernel(
    se::Stream* stream, const LaunchDimensions& launch_dimensions,
    PrimitiveType element_type,
    absl::Span<const se::DeviceMemoryBase> remote_input_buffers,
    se::DeviceMemoryBase local_input_buffer, se::DeviceMemoryBase output_buffer,
    RankId rank, int64_t num_ranks, int64_t num_elements,
    absl::Span<const se::DeviceMemoryBase> signal_flags_buffers);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_H_
