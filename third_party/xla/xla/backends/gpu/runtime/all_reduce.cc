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

#include "xla/backends/gpu/runtime/all_reduce.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

namespace {

using se::gpu::AllReduceStrategy;
static constexpr int64_t kKB = 1024;
static constexpr int64_t kMB = kKB * 1024;
static constexpr int64_t kMaxOneShotAllReduceSizeBytes = 2 * kMB;
static constexpr int64_t kMaxTwoShotAllReduceSizeBytes = 32 * kMB;

bool IsElementReductionSupported(PrimitiveType element_type,
                                 ReductionKind reduction_kind) {
  absl::Span<const HloOpcode> supported_ops =
      SupportedReductionOps(element_type);
  HloOpcode opcode = ReductionKindToOpcode(reduction_kind, element_type);
  return absl::c_linear_search(supported_ops, opcode);
}

}  // namespace

absl::Span<const HloOpcode> SupportedReductionOps(PrimitiveType element_type) {
  const auto numeric_types = llvm::concat<const PrimitiveType>(
      kSupportedFloatingPointTypes, kSupportedIntegralTypes);
  if (absl::c_linear_search(numeric_types, element_type)) {
    return absl::MakeSpan(gpu::kSupportedNumericReductionOps);
  }
  if (absl::c_linear_search(kSupportedPredicateTypes, element_type)) {
    return absl::MakeSpan(gpu::kSupportedLogicalReductionOps);
  }
  return {};
}

AllReduceStrategy GetAllReduceStrategy(int64_t input_size_bytes,
                                       int64_t world_size,
                                       bool is_multimem_enabled) {
  const int64_t read_size_bytes = input_size_bytes * world_size;
  if (read_size_bytes > kMaxOneShotAllReduceSizeBytes) {
    return AllReduceStrategy::kTwoShot;
  }
  if (is_multimem_enabled) {
    return AllReduceStrategy::kMultimem;
  }
  return AllReduceStrategy::kOneShot;
}

int64_t GetMaxSupportedAllReduceSizeBytes(AllReduceStrategy strategy) {
  switch (strategy) {
    case AllReduceStrategy::kOneShot:
      return kMaxOneShotAllReduceSizeBytes;
    case AllReduceStrategy::kTwoShot:
      return kMaxTwoShotAllReduceSizeBytes;
    case AllReduceStrategy::kMultimem:
      return kMaxTwoShotAllReduceSizeBytes;
  }
}

// A broader check overload for all-reduce kernel support.
// Returns absl::OkStatus() if supported, or an error status detailing why
// it is not supported.
absl::Status IsAllReduceKernelSupported(
    const se::DeviceDescription& device_info,        //
    int32_t num_operands,                            //
    std::optional<ReductionKind> reduction_kind,     //
    int64_t num_devices,                             //
    int64_t num_elements,                            //
    PrimitiveType element_type,                      //
    bool is_local,                                   //
    bool is_multimem_enabled,                        //
    const std::vector<ReplicaGroup>& replica_groups  //
) {
  // Check if the device supports Triton collective codegen:
  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  if (!device_info.cuda_compute_capability().IsAtLeastHopper() &&
      !device_info.gpu_compute_capability().IsRocm()) {
    return absl::UnimplementedError(absl::StrCat(
        "Triton collective codegen requires CUDA compute capability >= 9.0 "
        "(Hopper or newer) or a supported ROCm device. "
        "Got: ",
        device_info.gpu_compute_capability().ToString(), "."));
  }
  // ROCm: the cross-device barrier relies on a system-scope release store
  // becoming visible to peers without extra cache maintenance, which gfx90a
  // does not guarantee.
  if (const auto* rocm_cc =
          device_info.gpu_compute_capability().rocm_compute_capability();
      rocm_cc != nullptr && !rocm_cc->has_peer_visible_atomics()) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernels are not supported on ",
                     rocm_cc->gfx_version(), "."));
  }
  // TODO(b/383125489): Support variadic arguments.
  if (num_operands != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernel is not supported for number of "
                     "operands not equal to 1. Got ",
                     num_operands, "."));
  }
  if (replica_groups.empty()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  // The kernel bakes world_size (from the first replica group) at codegen and
  // sizes its buffers the same way, so it cannot serve cliques of different
  // sizes. Fall back to the library collective for non-uniform replica groups.
  for (const ReplicaGroup& group : replica_groups) {
    if (group.replica_ids_size() != num_devices) {
      return absl::UnimplementedError(absl::StrCat(
          "Collective kernel requires all replica groups to have the same "
          "size. Got a group of size ",
          group.replica_ids_size(), " but expected ", num_devices, "."));
    }
  }
  if (!reduction_kind.has_value()) {
    return absl::UnimplementedError("Could not match reduction computation.");
  }
  if (!is_local) {
    return absl::UnimplementedError(
        "Cross-host symmetric memory collectives are not supported.");
  }
  if (num_devices == 1) {
    return absl::UnimplementedError(
        "Degenerate collectives with a single device per replica group are not "
        "supported by collective kernels.");
  }
  if (!llvm::has_single_bit(static_cast<uint64_t>(num_devices))) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernels are only supported for power of 2 "
                     "number of devices. Got ",
                     num_devices, "."));
  }
  const int64_t byte_size =
      num_elements * primitive_util::ByteWidth(element_type);
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(byte_size, num_devices, is_multimem_enabled);
  if (byte_size * num_devices > GetMaxSupportedAllReduceSizeBytes(strategy)) {
    return absl::UnimplementedError(
        "Custom all-reduce strategy is only supported for small inputs.");
  }
  if (!IsElementReductionSupported(element_type, *reduction_kind)) {
    return absl::UnimplementedError(absl::StrCat(
        "Element type (",
        primitive_util::LowercasePrimitiveTypeName(element_type),
        ") and reduction kind (", absl::StrFormat("%v", *reduction_kind),
        ") combination is not supported."));
  }
  if (num_devices > stream_executor::gpu::kMaxNumAllReduceInputPtrs) {
    return absl::UnimplementedError(
        absl::StrCat("AllReduce XLA implementation does not support more than ",
                     se::gpu::kMaxNumAllReduceInputPtrs,
                     " ranks. Required number of ranks: ", num_devices));
  }
  return absl::OkStatus();
}

absl::StatusOr<AllReduceInfo> BuildAllReduceInfo(
    bool is_multimem_enabled, const GpuTopology& gpu_topology,
    const HloAllReduceInstruction* all_reduce,
    const DeviceAssignment* device_assignment) {
  if (!gpu_topology.has_gpu_target_config()) {
    return absl::InvalidArgumentError(
        "GpuTopology must have a target config to build AllReduceInfo.");
  }
  const se::DeviceDescription& device_info =
      gpu_topology.gpu_target_config().device_description;
  const int64_t num_devices =
      all_reduce->device_list()->num_devices_per_group();
  const std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(all_reduce->called_computations().front());
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_reduce->operand(0)->shape());
  const PrimitiveType element_type =
      all_reduce->operand(0)->shape().element_type();
  const int32_t num_operands = all_reduce->operand_count();
  const int64_t byte_size =
      num_elements * primitive_util::ByteWidth(element_type);
  const AllReduceStrategy strategy =
      GetAllReduceStrategy(byte_size, num_devices, is_multimem_enabled);
  ABSL_ASSIGN_OR_RETURN(
      const bool is_local,
      AreAllReplicasOnSameSlice(gpu_topology, *all_reduce, device_assignment));
  if (device_info.device_interconnect_info().active_links <= 0) {
    return absl::UnimplementedError(
        "Collective kernels are only supported on devices with NVLink/UALink "
        "support.");
  }
  ABSL_RETURN_IF_ERROR(IsAllReduceKernelSupported(device_info,                  //
                                             num_operands,                 //
                                             reduction_kind,               //
                                             num_devices,                  //
                                             num_elements,                 //
                                             element_type,                 //
                                             is_local,                     //
                                             is_multimem_enabled,          //
                                             all_reduce->replica_groups()  //
                                             ));
  return AllReduceInfo{
      /*.reduction_kind=*/*reduction_kind,
      /*.num_devices =*/num_devices,
      /*.num_elements =*/num_elements,
      /*.element_type =*/element_type,
      /*.all_reduce_strategy =*/strategy,
  };
}

absl::StatusOr<CollectiveKernelSpec> CreateAllReduceKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type) {
  int64_t group_size = instr->GetModule()->config().replica_count();
  if (!instr->replica_groups().empty() &&
      instr->replica_groups()[0].replica_ids_size() > 0) {
    group_size = instr->replica_groups()[0].replica_ids_size();
  }
  const int64_t input_size_bytes =
      ShapeUtil::ByteSizeOf(instr->operand(0)->shape());
  const se::gpu::AllReduceStrategy strategy = GetAllReduceStrategy(
      input_size_bytes, group_size, /*is_multimem_enabled=*/false);
  const int64_t num_signal_flags = group_size * launch_dimensions.num_blocks();
  const int64_t signal_size = xla::RoundUpTo<uint64_t>(
      num_signal_flags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);
  const int64_t remote_size =
      xla::RoundUpTo<uint64_t>(input_size_bytes, kXlaAllocatedBufferAlignBytes);

  CollectiveKernelSpec kernel_spec = {
      /* .codegen_config= */ {
          /* .copy_input_to_scratch= */ false,
          /* .input_buffer_specs= */
          {{/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
          /* .output_buffer_specs= */
          {{/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
          /* .argument_descriptors= */
          {{KernelArgType::kInputBuffer,
            /*index=*/0},  // buffers[0].source_buffer
           {KernelArgType::kOutputBuffer,
            /*index=*/0},  // buffers[0].dst_buffer
           {KernelArgType::kRuntimeRank},
           {KernelArgType::kInvocationCount},
           {KernelArgType::kScratchBuffer, /*index=*/0},   // signal buffers
           {KernelArgType::kScratchBuffer, /*index=*/1}},  // scratch buffers
          /* .sync_count_increment = */ 1 + static_cast<uint32_t>(strategy)},
      /* .scratch_buffers= */
      {{signal_size, /*requires_multimem=*/false,  // Signal buffers
        scratch_memory_type,
        /*should_memzero=*/true,
        /*should_double_buffer=*/true},
       {remote_size, /*requires_multimem=*/false,  // Remote buffers
        scratch_memory_type,
        /*should_memzero=*/false,
        /*should_double_buffer=*/true}}};
  return kernel_spec;
}

}  // namespace xla::gpu
