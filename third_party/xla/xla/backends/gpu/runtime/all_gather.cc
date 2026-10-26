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

#include "xla/backends/gpu/runtime/all_gather.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/bit.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

std::vector<PrimitiveType> SupportedAllGatherTypes(
    const se::GpuComputeCapability& gpu_cc) {
  std::vector<PrimitiveType> types;
  for (int i = PrimitiveType_MIN; i <= PrimitiveType_MAX; ++i) {
    if (!PrimitiveType_IsValid(i)) {
      continue;
    }
    const PrimitiveType type = static_cast<PrimitiveType>(i);
    if (!primitive_util::IsSubByteNonPredType(type) &&
        IsTritonSupportedDataType(type, gpu_cc)) {
      types.push_back(type);
    }
  }
  return types;
}

// A broader check for all-gather kernel support that verifies device, replica
// group, gather dimension, and element-type constraints.
// Returns absl::OkStatus() if supported, or an absl::UnimplementedError
// explaining why not.
absl::Status IsAllGatherKernelSupported(
    const se::DeviceDescription& device_info, int64_t num_devices,
    int64_t num_output_elements, int64_t per_rank_gather_dim_size,
    PrimitiveType element_type, bool is_local,
    const std::vector<ReplicaGroup>& replica_groups) {
  // Check if the device supports Triton collective codegen:
  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  if (!device_info.cuda_compute_capability().IsAtLeastHopper() &&
      !device_info.gpu_compute_capability().IsRocm()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Triton collective codegen requires CUDA compute capability >= 9.0 "
        "(Hopper or newer) or a supported ROCm device. Got: %s.",
        device_info.gpu_compute_capability().ToString()));
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
  if (replica_groups.empty()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
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
    return absl::UnimplementedError(absl::StrFormat(
        "Collective kernels are only supported for power of 2 number of "
        "devices. Got %d.",
        num_devices));
  }
  // Triton pads tile sizes to powers of 2, so each rank's slice along the
  // gather dimension must be a power of 2.
  if (!llvm::has_single_bit(static_cast<uint64_t>(per_rank_gather_dim_size))) {
    return absl::UnimplementedError(absl::StrFormat(
        "All-gather kernel requires the per-rank size along the gather "
        "dimension to be a power of 2. Got %d.",
        per_rank_gather_dim_size));
  }
  // Only types in SupportedAllGatherTypes() are allowed. Complex types, tokens,
  // tuples, and sub-byte types are not supported.
  if (!absl::c_linear_search(
          SupportedAllGatherTypes(device_info.gpu_compute_capability()),
          element_type)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Element type %s is not supported for the all-gather kernel. "
        "Supported types are predicates, 8/16/32/64-bit integers, and "
        "standard/8-bit floating-point types; use NCCL/RCCL for other types.",
        primitive_util::LowercasePrimitiveTypeName(element_type)));
  }

  const int64_t output_byte_size =
      num_output_elements * primitive_util::ByteWidth(element_type);
  if (output_byte_size > kMaxAllGatherSizeBytes) {
    return absl::UnimplementedError(
        "Custom all-gather strategy is only supported for small outputs.");
  }

  return absl::OkStatus();
}

absl::StatusOr<AllGatherInfo> BuildAllGatherInfo(
    const GpuTopology& gpu_topology, const HloAllGatherInstruction* all_gather,
    const DeviceAssignment* device_assignment) {
  if (!gpu_topology.has_gpu_target_config()) {
    return absl::InvalidArgumentError(
        "GpuTopology must have a target config to build AllGatherInfo.");
  }
  const se::DeviceDescription& device_info =
      gpu_topology.gpu_target_config().device_description;
  if (!all_gather->device_list()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  const int64_t num_devices =
      all_gather->device_list()->num_devices_per_group();
  if (all_gather->operand_count() > kAllGatherMaxNumOperands) {
    return absl::UnimplementedError(absl::StrFormat(
        "All-gather kernel supports at most %d operands. Got %d.",
        kAllGatherMaxNumOperands, all_gather->operand_count()));
  }
  if (device_info.device_interconnect_info().active_links <= 0) {
    return absl::UnimplementedError(
        "Collective kernels are only supported on devices with NVLink/UALink "
        "support.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const bool is_local,
      AreAllReplicasOnSameSlice(gpu_topology, *all_gather, device_assignment));

  int64_t max_output_elements = 0;
  for (const HloInstruction* operand : all_gather->operands()) {
    const int64_t output_elements =
        ShapeUtil::ElementsIn(operand->shape()) * num_devices;
    const int64_t per_rank_gather_dim_size =
        operand->shape().dimensions(all_gather->all_gather_dimension());
    ABSL_RETURN_IF_ERROR(IsAllGatherKernelSupported(
        device_info, num_devices, output_elements, per_rank_gather_dim_size,
        operand->shape().element_type(), is_local,
        all_gather->replica_groups()));
    max_output_elements = std::max(max_output_elements, output_elements);
  }
  return AllGatherInfo{
      /*.num_devices =*/num_devices,
      /*.max_output_elements =*/max_output_elements,
  };
}

absl::StatusOr<CollectiveKernelSpec> CreateAllGatherKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type) {
  int64_t group_size = instr->GetModule()->config().replica_count();
  if (!instr->replica_groups().empty() &&
      instr->replica_groups()[0].replica_ids_size() > 0) {
    group_size = instr->replica_groups()[0].replica_ids_size();
  }
  const int64_t num_operands = instr->operand_count();
  TF_RET_CHECK(num_operands >= 1) << "all-gather requires at least one operand";

  const int64_t num_signal_flags = group_size * launch_dimensions.num_blocks();
  const int64_t signal_size = xla::RoundUpTo<uint64_t>(
      num_signal_flags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);

  CollectiveKernelSpec kernel_spec;
  CollectiveCodegenConfig& codegen_config = kernel_spec.codegen_config;
  codegen_config.copy_input_to_scratch = false;
  codegen_config.sync_count_increment = 1u;
  // Argument layout (must match the Triton entry function): inputs, outputs,
  // rank, invocation count, signal flags, one remote scratch table per operand.
  for (int32_t i = 0; i < num_operands; ++i) {
    codegen_config.input_buffer_specs.push_back(
        {/*requires_multimem=*/false, SymmetricMemoryType::kNone});
    codegen_config.argument_descriptors.push_back(
        {KernelArgType::kInputBuffer, /*index=*/i});
  }
  for (int32_t i = 0; i < num_operands; ++i) {
    codegen_config.output_buffer_specs.push_back(
        {/*requires_multimem=*/false, SymmetricMemoryType::kNone});
    codegen_config.argument_descriptors.push_back(
        {KernelArgType::kOutputBuffer, /*index=*/i});
  }
  codegen_config.argument_descriptors.push_back({KernelArgType::kRuntimeRank});
  codegen_config.argument_descriptors.push_back(
      {KernelArgType::kInvocationCount});
  // Signal buffers.
  codegen_config.argument_descriptors.push_back(
      {KernelArgType::kScratchBuffer, /*index=*/0});
  kernel_spec.scratch_buffers.push_back({signal_size,
                                         /*requires_multimem=*/false,
                                         scratch_memory_type,
                                         /*should_memzero=*/true,
                                         /*should_double_buffer=*/true});
  // One symmetric scratch buffer per operand that holds one rank's slice
  // (= input size). Double-buffering is handled by the should_double_buffer
  // flag.
  for (int32_t i = 0; i < num_operands; ++i) {
    const int64_t remote_size = xla::RoundUpTo<uint64_t>(
        ShapeUtil::ByteSizeOf(instr->operand(i)->shape()),
        kXlaAllocatedBufferAlignBytes);
    codegen_config.argument_descriptors.push_back(
        {KernelArgType::kScratchBuffer, /*index=*/1 + i});
    kernel_spec.scratch_buffers.push_back({remote_size,
                                           /*requires_multimem=*/false,
                                           scratch_memory_type,
                                           /*should_memzero=*/false,
                                           /*should_double_buffer=*/true});
  }
  return kernel_spec;
}

}  // namespace xla::gpu
