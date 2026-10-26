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

#include "xla/backends/gpu/runtime/all_to_all.h"

#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/bit.h"
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
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

absl::Status IsAllToAllKernelSupported(
    const se::DeviceDescription& device_info, int32_t num_operands,
    bool has_split_dimension, int64_t num_devices, int64_t num_elements,
    int64_t per_rank_split_dim_size, PrimitiveType element_type, bool is_local,
    const std::vector<ReplicaGroup>& replica_groups) {
  // Default threshold for custom all-to-all in bytes of total transported data
  // across all participating devices for the collective kernel. Divided by the
  // number of participating devices when compared against per-rank input/output
  // buffer size.
  static constexpr int64_t kMaxAllToAllSizeBytes =
      256LL * 1024 * 1024;  // 256 MB

  // Check if the device supports Triton collective codegen:
  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  if (!device_info.cuda_compute_capability().IsAtLeastHopper() &&
      !device_info.gpu_compute_capability().IsRocm()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Triton collective codegen requires CUDA compute capability >= 9.0 "
        "(Hopper or newer) or a supported ROCm device. Got: %s.",
        device_info.gpu_compute_capability().ToString()));
  }

  if (const auto* rocm_cc =
          device_info.gpu_compute_capability().rocm_compute_capability();
      rocm_cc != nullptr && !rocm_cc->has_peer_visible_atomics()) {
    return absl::UnimplementedError(
        absl::StrCat("Collective kernels are not supported on ",
                     rocm_cc->gfx_version(), "."));
  }
  if (num_operands != 1) {
    return absl::UnimplementedError(absl::StrFormat(
        "Collective kernel is not supported for number of operands not equal "
        "to 1. Got %d.",
        num_operands));
  }
  if (!has_split_dimension) {
    return absl::UnimplementedError(
        "All-to-all kernel requires a split dimension (array all-to-all).");
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
  if (per_rank_split_dim_size <= 0 ||
      !llvm::has_single_bit(static_cast<uint64_t>(per_rank_split_dim_size))) {
    return absl::UnimplementedError(absl::StrFormat(
        "All-to-all kernel requires the per-rank size along the split "
        "dimension to be a power of 2. Got %d.",
        per_rank_split_dim_size));
  }
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Sub-byte element type %s is not supported for the all-to-all kernel.",
        primitive_util::LowercasePrimitiveTypeName(element_type)));
  }

  const int64_t output_byte_size =
      num_elements * primitive_util::ByteWidth(element_type);
  // The threshold is per total transported data, therefore it is divided by the
  // number of participating devices.
  if (output_byte_size > kMaxAllToAllSizeBytes / num_devices) {
    return absl::UnimplementedError(
        "Custom all-to-all strategy is only supported for small outputs.");
  }

  return absl::OkStatus();
}

}  // namespace

AllToAllStrategy GetAllToAllStrategy(int64_t input_size_bytes,
                                     int64_t world_size) {
  // Maximum total transported data size in bytes across all participating
  // devices for the one-shot all-to-all strategy (1-barrier double-buffered
  // scratch push). Divided by the number of participating devices when compared
  // against per-rank input/output buffer size.
  static constexpr int64_t kMaxOneShotAllToAllSizeBytes =
      32LL * 1024 * 1024;  // 32 MB
  if (input_size_bytes <= kMaxOneShotAllToAllSizeBytes / world_size) {
    return AllToAllStrategy::kOneShot;
  }
  return AllToAllStrategy::kTwoShot;
}

absl::StatusOr<AllToAllInfo> BuildAllToAllInfo(
    const GpuTopology& gpu_topology, const HloAllToAllInstruction* all_to_all,
    const DeviceAssignment* device_assignment) {
  if (!gpu_topology.has_gpu_target_config()) {
    return absl::InvalidArgumentError(
        "GpuTopology must have a target config to build AllToAllInfo.");
  }
  const se::DeviceDescription& device_info =
      gpu_topology.gpu_target_config().device_description;
  if (!all_to_all->device_list()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  const int64_t num_devices =
      all_to_all->device_list()->num_devices_per_group();
  const int32_t num_operands = all_to_all->operand_count();
  const bool has_split_dimension = all_to_all->split_dimension().has_value() &&
                                   all_to_all->shape().IsArray();
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_to_all->operand(0)->shape());
  int64_t per_rank_split_dim_size = 0;
  if (has_split_dimension && num_devices > 0) {
    const int64_t split_dim_size = all_to_all->operand(0)->shape().dimensions(
        *all_to_all->split_dimension());
    if (split_dim_size % num_devices == 0) {
      per_rank_split_dim_size = split_dim_size / num_devices;
    }
  }
  const PrimitiveType element_type =
      all_to_all->operand(0)->shape().element_type();
  if (device_info.device_interconnect_info().active_links <= 0) {
    return absl::UnimplementedError(
        "Collective kernels are only supported on devices with NVLink/UALink "
        "support.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const bool is_local,
      AreAllReplicasOnSameSlice(gpu_topology, *all_to_all, device_assignment));
  ABSL_RETURN_IF_ERROR(IsAllToAllKernelSupported(
      device_info, num_operands, has_split_dimension, num_devices, num_elements,
      per_rank_split_dim_size, element_type, is_local,
      all_to_all->replica_groups()));
  const int64_t byte_size =
      num_elements * primitive_util::ByteWidth(element_type);
  return AllToAllInfo{
      /*.num_devices =*/num_devices,
      /*.num_elements =*/num_elements,
      /*.element_type =*/element_type,
      /*.all_to_all_strategy =*/GetAllToAllStrategy(byte_size, num_devices),
  };
}

absl::StatusOr<CollectiveKernelSpec> CreateAllToAllKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type) {
  int64_t group_size = instr->GetModule()->config().replica_count();
  if (!instr->replica_groups().empty() &&
      instr->replica_groups()[0].replica_ids_size() > 0) {
    group_size = instr->replica_groups()[0].replica_ids_size();
  }

  const int64_t num_signal_flags = group_size * launch_dimensions.num_blocks();
  const int64_t signal_size = xla::RoundUpTo<uint64_t>(
      num_signal_flags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);

  const int64_t input_size_bytes =
      ShapeUtil::ByteSizeOf(instr->operand(0)->shape());
  const bool is_one_shot = GetAllToAllStrategy(input_size_bytes, group_size) ==
                           AllToAllStrategy::kOneShot;
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
          {{KernelArgType::kInputBuffer, /*index=*/0},
           {KernelArgType::kOutputBuffer, /*index=*/0},
           {KernelArgType::kRuntimeRank},
           {KernelArgType::kInvocationCount},
           {KernelArgType::kScratchBuffer,
            /*index=*/0},  // signal buffers
           {KernelArgType::kScratchBuffer,
            /*index=*/1}},  // remote scratch buffers
          /* .sync_count_increment= */ is_one_shot ? 1u : 2u},
      /* .scratch_buffers= */
      {{signal_size, /*requires_multimem=*/false, scratch_memory_type,
        /*should_memzero=*/true,
        /*should_double_buffer=*/true},
       {remote_size, /*requires_multimem=*/false, scratch_memory_type,
        /*should_memzero=*/false,
        /*should_double_buffer=*/is_one_shot}}};
  return kernel_spec;
}

}  // namespace xla::gpu
