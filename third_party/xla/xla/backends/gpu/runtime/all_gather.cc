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
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Alignment.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/computation_placer.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

absl::Status IsAllGatherKernelSupported(int64_t num_elements,
                                        PrimitiveType element_type) {
  // Only types in kSupportedAllGatherTypes are allowed. Triton tt.load/tt.store
  // support signless integers and floating-point types; unsigned integers,
  // complex types, tokens, tuples, and exotic types (e.g. 4-bit, 8-bit floats)
  // are not supported.
  if (!absl::c_linear_search(kSupportedAllGatherTypes, element_type)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Element type %s is not supported for the all-gather kernel. "
        "Supported types are signed integers and standard floating-point "
        "types; use NCCL/RCCL for other types.",
        primitive_util::LowercasePrimitiveTypeName(element_type)));
  }

  // The total transfer size in bits must be aligned to
  // kBitsPerMemoryTransaction (128 bits = 16 bytes) so each thread can
  // load/store a complete transaction.
  const uint64_t element_bits = primitive_util::BitWidth(element_type);
  if (!llvm::isAligned(llvm::Align(kBitsPerMemoryTransaction),
                       static_cast<uint64_t>(num_elements) * element_bits)) {
    return absl::UnimplementedError(absl::StrFormat(
        "Number of elements (%d) of type %s (%d bits each) is not aligned to "
        "the memory transaction alignment requirement (%d bits).",
        num_elements, primitive_util::LowercasePrimitiveTypeName(element_type),
        element_bits, kBitsPerMemoryTransaction));
  }
  return absl::OkStatus();
}

absl::Status IsAllGatherKernelSupported(
    bool is_collective_kernel_enabled, const se::DeviceDescription& device_info,
    int32_t num_operands, int64_t num_devices, int64_t num_elements,
    PrimitiveType element_type, bool is_local,
    const std::vector<ReplicaGroup>& replica_groups) {
  if (!is_collective_kernel_enabled) {
    return absl::UnimplementedError("Collective kernel is not enabled.");
  }
  // Check if the device supports Triton collective codegen:
  // CUDA: Requires compute capability 9.0+ (Hopper or newer)
  // ROCm: All versions with Triton support are enabled
  if (!device_info.cuda_compute_capability().IsAtLeastHopper() &&
      !device_info.gpu_compute_capability().IsRocm()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Triton collective codegen requires CUDA compute capability >= 9.0 "
        "(Hopper or newer) or a ROCm device with Triton support. Got: %s.",
        device_info.gpu_compute_capability().ToString()));
  }
  // TODO(b/383125489): Support variadic arguments.
  if (num_operands != 1) {
    return absl::UnimplementedError(absl::StrFormat(
        "Collective kernel is not supported for number of operands not equal "
        "to 1. Got %d.",
        num_operands));
  }
  if (replica_groups.empty()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  if (!is_local) {
    return absl::UnimplementedError(
        "Cross-host symmetric memory collectives are not supported.");
  }
  if (!llvm::has_single_bit(static_cast<uint64_t>(num_devices))) {
    return absl::UnimplementedError(absl::StrFormat(
        "Collective kernels are only supported for power of 2 number of "
        "devices. Got %d.",
        num_devices));
  }
  return IsAllGatherKernelSupported(num_elements, element_type);
}

absl::StatusOr<AllGatherInfo> BuildAllGatherInfo(
    bool is_collective_kernel_enabled, const GpuTopology& gpu_topology,
    const HloAllGatherInstruction* all_gather,
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
  const int64_t num_elements =
      ShapeUtil::ElementsIn(all_gather->operand(0)->shape());
  const PrimitiveType element_type =
      all_gather->operand(0)->shape().element_type();
  const int32_t num_operands = all_gather->operand_count();
  if (device_info.device_interconnect_info().active_links <= 0) {
    return absl::UnimplementedError(
        "Collective kernels are only supported on devices with NVLink/UALink "
        "support.");
  }
  ASSIGN_OR_RETURN(const CollectiveOpGroupMode group_mode,
                   GetCollectiveOpGroupMode(all_gather));
  const bool is_local = IsAllReplicasLocal(
      gpu_topology.num_devices_per_process(), all_gather->replica_groups(),
      group_mode, device_assignment);
  RETURN_IF_ERROR(IsAllGatherKernelSupported(
      is_collective_kernel_enabled, device_info, num_operands, num_devices,
      num_elements, element_type, is_local, all_gather->replica_groups()));
  return AllGatherInfo{
      /*.num_devices =*/num_devices,
      /*.num_elements =*/num_elements,
      /*.element_type =*/element_type,
  };
}

namespace {
// All-gather always uses a one-shot strategy: every rank reads its peers'
// slices directly, so the work is proportional to the full element count
// with no rank-based partitioning.
static constexpr uint64_t kMaxThreadsPerBlock = 512;
}  // namespace

LaunchDimensions AllGatherLaunchDimensions(int64_t elements,
                                           int64_t warp_size) {
  // Maximum number of threads such that each thread has elements to process.
  // Round up to a multiple of warp_size so every warp is fully occupied.
  const int64_t total_threads = RoundUpTo(
      CeilOfRatio(elements, se::gpu::kNumElementsPerThread), warp_size);
  // Triton expects power of 2 for threads_per_block / threads_per_warp.
  const int64_t threads_per_block =
      std::min(kMaxThreadsPerBlock,
               llvm::bit_ceil(static_cast<uint64_t>(total_threads)));
  const int64_t blocks_per_grid =
      std::min(kAllGatherMaxBlocksPerGrid,
               CeilOfRatio(total_threads, threads_per_block));
  return LaunchDimensions(blocks_per_grid, threads_per_block);
}

absl::StatusOr<CollectiveKernelSpec> CreateAllGatherKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions) {
  // instr may be the raw kAllGather instruction or a fusion wrapping it.
  const HloInstruction* all_gather = instr;
  if (instr->opcode() == HloOpcode::kFusion) {
    all_gather = instr->fused_instructions_computation()->root_instruction();
  }

  int64_t group_size = instr->GetModule()->config().replica_count();
  if (!all_gather->replica_groups().empty() &&
      all_gather->replica_groups()[0].replica_ids_size() > 0) {
    group_size = all_gather->replica_groups()[0].replica_ids_size();
  }

  // The symmetric scratch buffer holds one rank's slice (= input size).
  // Double-buffering is handled by the should_double_buffer flag.
  const int64_t input_size_bytes =
      ShapeUtil::ByteSizeOf(all_gather->operand(0)->shape());
  const int64_t num_signal_flags = group_size * launch_dimensions.num_blocks();
  const int64_t signal_size = xla::RoundUpTo<uint64_t>(
      num_signal_flags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);
  // Each rank allocates its own input-sized symmetric buffer; pointers are
  // exchanged via kXlaRendezvous so every rank can read from each peer.
  const int64_t remote_size =
      xla::RoundUpTo<uint64_t>(input_size_bytes, kXlaAllocatedBufferAlignBytes);

  CollectiveKernelSpec kernel_spec = {
      /* .input_buffer_specs= */ {
          {/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
      /* .output_buffer_specs= */
      {{/*requires_multimem=*/false, SymmetricMemoryType::kNone}},
      /* .scratch_buffers= */
      {{signal_size, /*requires_multimem=*/false,  // Signal flags
        SymmetricMemoryType::kXlaRendezvous,
        /*should_memzero=*/true,
        /*should_double_buffer=*/true},
       {remote_size, /*requires_multimem=*/false,  // Symmetric remote buffer
        SymmetricMemoryType::kXlaRendezvous,
        /*should_memzero=*/false,
        /*should_double_buffer=*/true}},
      /* .argument_descriptors= */
      {{KernelArgType::kInputBuffer, /*index=*/0},   // buffers[0].source_buffer
       {KernelArgType::kOutputBuffer, /*index=*/0},  // buffers[0].dst_buffer
       {KernelArgType::kRuntimeRank},
       {KernelArgType::kInvocationCount},
       {KernelArgType::kScratchBuffer, /*index=*/0},   // signal buffers
       {KernelArgType::kScratchBuffer, /*index=*/1}},  // remote buffers
      /* .sync_count_increment= */ 1u};  // AllGather is always one-shot
  return kernel_spec;
}

}  // namespace xla::gpu
