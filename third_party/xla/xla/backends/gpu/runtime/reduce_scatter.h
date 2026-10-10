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

#ifndef XLA_BACKENDS_GPU_RUNTIME_REDUCE_SCATTER_H_
#define XLA_BACKENDS_GPU_RUNTIME_REDUCE_SCATTER_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Maximum size of the reduce-scatter *input* (per device, before the scatter)
// for which the Triton reduce-scatter kernel is used.
inline constexpr int64_t kMaxReduceScatterSizeBytes =
    16 * 1024 * 1024;  // 16 MB

// Maximum number of blocks per grid for the Triton reduce-scatter kernel.
inline constexpr int64_t kReduceScatterMaxBlocksPerGrid = 64;

// Encapsulates the information needed to perform a reduce-scatter via the
// Triton collective kernel backend.
struct ReduceScatterInfo {
  int64_t num_devices;
  // Number of elements of the output (per device, after the scatter).
  int64_t num_output_elements;
};

// Returns true if the reduce-scatter can be flattened to a 1D reduce-scatter
// along dimension 0, i.e. the scatter dimension is the physically major-most
// non-trivial dimension of the operand and the result has the same layout as
// the operand. In that case, the chunk sent to rank `i` is the contiguous
// range [i * O, (i + 1) * O) of the operand where O is the number of elements
// in the result.
bool IsReduceScatterFlattenable(const HloReduceScatterInstruction* rs);

// Constructs a ReduceScatterInfo object for the given reduce-scatter
// instruction. Returns absl::UnimplementedError if the Triton reduce-scatter
// kernel is not supported for this instruction's configuration.
absl::StatusOr<ReduceScatterInfo> BuildReduceScatterInfo(
    bool is_collective_kernel_enabled, const GpuTopology& gpu_topology,
    const HloReduceScatterInstruction* reduce_scatter,
    const DeviceAssignment* device_assignment);

// Returns the launch dimensions for the reduce-scatter kernel.
LaunchDimensions ReduceScatterLaunchDimensions(
    int64_t num_output_elements, int64_t num_ranks,
    const se::DeviceDescription& device_info);

// Creates a CollectiveKernelSpec describing the resource requirements of a
// Triton one-shot reduce-scatter kernel. The kernel argument layout is:
//   [0] input buffer  (kInputBuffer, index 0)
//   [1] output buffer (kOutputBuffer, index 0)
//   [2] runtime rank  (kRuntimeRank)
//   [3] signal flags (kScratchBuffer, index 0)
//   [4] remote input buffer pointer table (kScratchBuffer, index 1)
// `scratch_memory_type` is used for both scratch buffers.
absl::StatusOr<CollectiveKernelSpec> CreateReduceScatterKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_REDUCE_SCATTER_H_
