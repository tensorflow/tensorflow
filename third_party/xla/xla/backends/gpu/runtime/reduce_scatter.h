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

namespace xla::gpu {

// Maximum input byte size per operand for the custom reduce-scatter kernel.
inline constexpr int64_t kMaxReduceScatterSizeBytes =
    16 * 1024 * 1024;  // 16 MB

// Maximum number of blocks per grid for the Triton reduce-scatter kernel.
inline constexpr int64_t kReduceScatterMaxBlocksPerGrid = 64;

// Maximum number of operands supported by the variadic Triton reduce-scatter
// kernel.
inline constexpr int64_t kReduceScatterMaxNumOperands = 16;

// Encapsulates the information needed to perform a reduce-scatter via the
// Triton collective kernel backend.
struct ReduceScatterInfo {
  int64_t num_devices;
  // Number of elements of the largest output (per device, after the scatter)
  // across the operands of a (possibly variadic) reduce-scatter.
  int64_t num_output_elements;
};

// Returns true if every operand's scatter dimension is its most-major
// non-trivial physical dimension and its output has the same layout, so the
// part of the input reduced into rank `i` is the contiguous slice
// `[i * O_k, (i + 1) * O_k)`.
bool IsReduceScatterFlattenable(const HloReduceScatterInstruction* rs);

// Constructs a ReduceScatterInfo object for the given (possibly variadic)
// reduce-scatter instruction. Returns absl::UnimplementedError if the Triton
// reduce-scatter kernel is not supported for this instruction's configuration.
absl::StatusOr<ReduceScatterInfo> BuildReduceScatterInfo(
    const GpuTopology& gpu_topology,
    const HloReduceScatterInstruction* reduce_scatter,
    const DeviceAssignment* device_assignment);

// Creates a CollectiveKernelSpec describing the resource requirements of a
// Triton one-shot reduce-scatter kernel with N operands. The kernel argument
// layout is:
//   [0, N)       input buffers  (kInputBuffer, index k)
//   [N, 2N)      output buffers (kOutputBuffer, index k)
//   [2N]         runtime rank   (kRuntimeRank)
//   [2N + 1]     signal flags   (kScratchBuffer, index 0)
//   [2N + 2 + k] remote input buffer pointer table of operand k
//                (kScratchBuffer, index 1 + k)
// `scratch_memory_type` is used for all scratch buffers.
absl::StatusOr<CollectiveKernelSpec> CreateReduceScatterKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_REDUCE_SCATTER_H_
