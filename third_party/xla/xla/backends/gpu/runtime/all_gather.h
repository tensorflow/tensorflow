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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_H_

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns the element types supported by the Triton all-gather kernel: all
// Triton-supported data types that are not sub-byte types (except PRED).
std::vector<PrimitiveType> SupportedAllGatherTypes(
    const se::GpuComputeCapability& gpu_cc);

// Maximum number of GPU thread-blocks launched per all-gather kernel.
// This number is set by picking the max number of blocks for all reduce and
// scaling it because the outputs are bigger for all-gather.
// The number is somewhat arbitrary and should be revisited.
inline constexpr int64_t kAllGatherMaxBlocksPerGrid = 64;

// Optimal threshold for one-shot all-gather in bytes for the collective kernel,
// applied to the gathered output buffer size.
// Base on the experimental results.
inline constexpr int64_t kMaxAllGatherSizeBytes = 16 * 1024 * 1024;  // 16 MB

// Maximum number of operands supported by the variadic Triton all-gather
// kernel.
inline constexpr int64_t kAllGatherMaxNumOperands = 16;

// Encapsulates the information needed to perform an all-gather via the Triton
// collective kernel backend.
struct AllGatherInfo {
  int64_t num_devices;
  int64_t max_output_elements;
};

// Constructs an AllGatherInfo object for the given all-gather instruction.
// Returns absl::UnimplementedError if the Triton all-gather kernel is not
// supported for this instruction's configuration.
absl::StatusOr<AllGatherInfo> BuildAllGatherInfo(
    const GpuTopology& gpu_topology, const HloAllGatherInstruction* all_gather,
    const DeviceAssignment* device_assignment);

// Creates a CollectiveKernelSpec describing the resource requirements of a
// Triton all-gather kernel with N operands. The kernel argument layout matches
// all-reduce:
//   [0, N)      local input buffers (kInputBuffer, index i)
//   [N, 2N)     output buffers (kOutputBuffer, index i)
//   [2N]        runtime rank  (kRuntimeRank)
//   [2N + 1]    invocation count (kInvocationCount)
//   [2N + 2]    signal flags (kScratchBuffer, index 0)
//   [2N + 3 + i] remote scratch buffer pointer table of operand i
//               (kScratchBuffer, index 1 + i)
// `scratch_memory_type` is used for all scratch buffers.
absl::StatusOr<CollectiveKernelSpec> CreateAllGatherKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_H_
