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

#include <array>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Memory transaction width for the Triton all-gather kernel in bits (128 bits =
// 16 bytes). The total buffer size in bits must be aligned to this value so
// that each thread can load/store a complete 128-bit transaction.
inline constexpr uint64_t kBitsPerMemoryTransaction = 128;

// Element types supported by the Triton all-gather kernel.
// Triton tt.load/tt.store support signless integers and floating-point types.
// Unsigned integer types, complex types, tokens, tuples, and exotic types
// (e.g. 4-bit integers, 8-bit floats) are not supported.
inline constexpr auto kSupportedAllGatherTypes =
    std::array{F16, BF16, F32, F64, S8, S16, S32, S64};

// Maximum number of GPU thread-blocks launched per all-gather kernel.
// This constant is shared between the kernel launcher (all_gather.cc) and the
// unmanaged-argument shaper (collective_emitter.cc) so that the signal buffer
// is always sized to match the actual grid.
inline constexpr int64_t kAllGatherMaxBlocksPerGrid = 32;

// Encapsulates the information needed to perform an all-gather via the Triton
// collective kernel backend.
struct AllGatherInfo {
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
};

// Returns absl::OkStatus() if the all-gather kernel is supported for the given
// element type and number of elements, or an error status detailing why it is
// not supported.
absl::Status IsAllGatherKernelSupported(int64_t num_elements,
                                        PrimitiveType element_type);

// A broader check for all-gather kernel support that verifies device, operand
// count, replica group, and element-type constraints.
// Returns absl::OkStatus() if supported, or an absl::UnimplementedError
// explaining why not.
absl::Status IsAllGatherKernelSupported(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    int32_t num_operands, int64_t num_devices, int64_t num_elements,
    PrimitiveType element_type, bool is_local,
    const std::vector<ReplicaGroup>& replica_groups);

// Constructs an AllGatherInfo object for the given all-gather instruction.
// Returns absl::UnimplementedError if the Triton all-gather kernel is not
// supported for this instruction's configuration.
absl::StatusOr<AllGatherInfo> BuildAllGatherInfo(
    bool is_collective_kernel_enabled, const GpuTopology& gpu_topology,
    const HloAllGatherInstruction* all_gather,
    const DeviceAssignment* device_assignment);

// Returns the launch dimensions for the all-gather kernel.
// All-gather uses a one-shot strategy: each rank contributes its local slice
// to the symmetric buffer and then reads each peer's slice.
// warp_size should be device_description.threads_per_warp() (32 on NVIDIA,
// 64 on AMD) so that the thread count is a multiple of the hardware warp.
LaunchDimensions AllGatherLaunchDimensions(int64_t elements, int64_t warp_size);

// Creates a CollectiveKernelSpec describing the resource requirements of a
// Triton all-gather kernel.  The returned spec uses the same 6-argument layout
// as the all-reduce kernel:
//   [0] input buffer (per-rank source slice)
//   [1] output buffer (full gathered destination)
//   [2] runtime rank  (kRuntimeRank)
//   [3] invocation count (kInvocationCount)
//   [4] scratch index 0: signal flags (kScratchBuffer)
//   [5] scratch index 1: symmetric remote buffer (kScratchBuffer)
absl::StatusOr<CollectiveKernelSpec> CreateAllGatherKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_GATHER_H_
