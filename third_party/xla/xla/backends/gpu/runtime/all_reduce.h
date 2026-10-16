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

#include <array>
#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla {
class DeviceAssignment;
class GpuTopology;
}  // namespace xla

namespace xla::gpu {

// Encapsulates the information needed to perform an all-reduce.
struct AllReduceInfo {
  ReductionKind reduction_kind;
  int64_t num_devices;
  int64_t num_elements;
  PrimitiveType element_type;
  se::gpu::AllReduceStrategy all_reduce_strategy;
};

constexpr auto kSupportedFloatingPointTypes = std::array{F32, BF16, F64};
constexpr auto kSupportedIntegralTypes = std::array{S32, S64};
constexpr auto kSupportedPredicateTypes = std::array{PRED};
constexpr auto kSupportedNumericReductionOps =
    std::array{HloOpcode::kAdd, HloOpcode::kMultiply, HloOpcode::kMaximum,
               HloOpcode::kMinimum};
constexpr auto kSupportedLogicalReductionOps =
    std::array{HloOpcode::kOr, HloOpcode::kAnd};

// Returns the supported element types for all-reduce.
inline auto SupportedTypes() {
  return llvm::concat<const PrimitiveType>(gpu::kSupportedFloatingPointTypes,
                                           gpu::kSupportedIntegralTypes,
                                           gpu::kSupportedPredicateTypes);
}

// Returns the supported reduction ops for the given element type as a span.
absl::Span<const HloOpcode> SupportedReductionOps(PrimitiveType element_type);

// Returns the all-reduce strategy for the given input size and world size.
// If `is_multimem_enabled` is true, then multimem strategies are also
// considered.
se::gpu::AllReduceStrategy GetAllReduceStrategy(int64_t input_size_bytes,
                                                int64_t world_size,
                                                bool is_multimem_enabled);

// Returns the maximum supported all-reduce read size in bytes for the given
// strategy.
int64_t GetMaxSupportedAllReduceSizeBytes(se::gpu::AllReduceStrategy strategy);

// Creates a CollectiveKernelSpec for a given all-reduce instruction.
// `scratch_memory_type` is used for both scratch buffers.
absl::StatusOr<CollectiveKernelSpec> CreateAllReduceKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type);

// Constructs an AllReduceInfo object for the given all-reduce instruction.
// Returns an error status if the all-reduce kernel is not supported.
absl::StatusOr<AllReduceInfo> BuildAllReduceInfo(
    bool is_multimem_enabled, const GpuTopology& gpu_topology,
    const HloAllReduceInstruction* all_reduce,
    const DeviceAssignment* device_assignment);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_REDUCE_H_
