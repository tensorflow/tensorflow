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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_H_
#define XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns true if the kernel is supported for the given number of outputs and
// element type.
bool IsRaggedAllToAllKernelSupported(int64_t num_outputs,
                                     PrimitiveType element_type);

// Input:
//  - input_buffer: dtype[num_input_rows, num_row_elements]
//  - input_offsets_buffer: s64[num_ranks * num_updates_per_rank]
//  - send_sizes_buffer: s64[num_ranks * num_updates_per_rank]
//  - output_offsets_buffer: s64[num_ranks * num_updates_per_rank]
//  - num_outputs: number of output buffers
//  - num_updates_per_output: number of updates to write to each output buffer
//  - num_input_rows: number of input rows
//  - num_row_elements: number of elements in each row
// Output:
//  - output_buffers[num_outputs, num_output_rows, num_row_elements]
// Outputs are filled with the updates from the input data. Number of output
// rows is not explicitly specified in the API, but should be enough to fit all
// the inputs. It is the responsibility of the caller to make sure that it is
// the case.
absl::Status RunRaggedAllToAllKernel(
    se::Stream* stream, PrimitiveType element_type,
    se::DeviceMemoryBase input_buffer,
    absl::Span<const se::DeviceMemoryBase> output_buffers,
    se::DeviceMemoryBase input_offsets_buffer,
    se::DeviceMemoryBase send_sizes_buffer,
    se::DeviceMemoryBase output_offsets_buffer, int64_t num_outputs,
    int64_t num_updates_per_output, int64_t num_input_rows,
    int64_t num_row_elements);
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_H_
