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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Strategy for performing an all-to-all via the Triton collective kernel
// backend.
enum class AllToAllStrategy {
  // 1-barrier double-buffered scratch push.
  kOneShot,
  // 2-barrier single-buffered scratch push with entry + exit barriers.
  kTwoShot,
};

// Encapsulates the information needed to perform an all-to-all via the Triton
// collective kernel backend.
struct AllToAllInfo {
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
  AllToAllStrategy all_to_all_strategy = AllToAllStrategy::kOneShot;
};

// Returns the all-to-all strategy for the given per-rank input size in bytes
// and world size.
AllToAllStrategy GetAllToAllStrategy(int64_t input_size_bytes,
                                     int64_t world_size);

// Constructs an AllToAllInfo object for the given all-to-all instruction.
// Returns absl::UnimplementedError if the Triton all-to-all kernel is not
// supported for this instruction's configuration.
absl::StatusOr<AllToAllInfo> BuildAllToAllInfo(
    const GpuTopology& gpu_topology, const HloAllToAllInstruction* all_to_all,
    const DeviceAssignment* device_assignment);

// Creates a CollectiveKernelSpec describing the resource requirements of a
// Triton all-to-all kernel. The kernel argument layout matches all-gather:
//   [0] local input buffer (kInputBuffer, index 0)
//   [1] output buffer (kOutputBuffer, index 0)
//   [2] runtime rank  (kRuntimeRank)
//   [3] invocation count (kInvocationCount)
//   [4] signal flags (kScratchBuffer, index 0)
//   [5] remote scratch buffer pointer table (kScratchBuffer, index 1)
// `scratch_memory_type` is used for both scratch buffers.
absl::StatusOr<CollectiveKernelSpec> CreateAllToAllKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_H_
