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

#include "xla/backends/gpu/runtime/reduce_scatter.h"

#include <algorithm>
#include <cstdint>
#include <optional>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/bit.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu_topology.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

bool IsReduceScatterFlattenable(const HloReduceScatterInstruction* rs) {
  if (rs->operand_count() != 1 || !rs->shape().IsArray()) {
    VLOG(3) << "Reduce-scatter " << rs->name()
            << " is not flattenable: expected a single array operand.";
    return false;
  }
  const Shape& in = rs->operand(0)->shape();
  const Shape& out = rs->shape();
  if (!in.has_layout() || !out.has_layout() ||
      !LayoutUtil::Equal(in.layout(), out.layout())) {
    VLOG(3) << "Reduce-scatter " << rs->name()
            << " is not flattenable: input and output layouts do not match.";
    return false;
  }
  const int64_t scatter_dim = rs->scatter_dimension();
  // All dimensions that are physically more major than the scatter dimension
  // must be trivial.
  for (int64_t dim : llvm::reverse(in.layout().minor_to_major())) {
    if (dim == scatter_dim) {
      return true;
    }
    if (in.dimensions(dim) != 1) {
      VLOG(3) << "Reduce-scatter " << rs->name()
              << " is not flattenable: non-trivial dimension " << dim
              << " (size " << in.dimensions(dim)
              << ") is more major than scatter dimension " << scatter_dim
              << ".";
      return false;
    }
  }
  VLOG(3) << "Reduce-scatter " << rs->name()
          << " is not flattenable: scatter dimension " << scatter_dim
          << " not found in layout.";
  return false;
}

absl::StatusOr<ReduceScatterInfo> BuildReduceScatterInfo(
    bool is_collective_kernel_enabled, const GpuTopology& gpu_topology,
    const HloReduceScatterInstruction* reduce_scatter,
    const DeviceAssignment* device_assignment) {
  if (!is_collective_kernel_enabled) {
    return absl::UnimplementedError("Collective kernel is not enabled.");
  }
  if (!gpu_topology.has_gpu_target_config()) {
    return absl::InvalidArgumentError(
        "GpuTopology must have a target config to build ReduceScatterInfo.");
  }
  const se::DeviceDescription& device_info =
      gpu_topology.gpu_target_config().device_description;
  if (!device_info.cuda_compute_capability().IsAtLeastHopper()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Triton reduce-scatter requires CUDA compute capability >= 9.0. "
        "Got: %s.",
        device_info.gpu_compute_capability().ToString()));
  }
  if (device_info.device_interconnect_info().active_links <= 0) {
    return absl::UnimplementedError(
        "Collective kernels are only supported on devices with NVLink "
        "support.");
  }
  if (reduce_scatter->operand_count() != 1) {
    return absl::UnimplementedError(
        "Reduce-scatter kernel only supports a single operand.");
  }
  if (!reduce_scatter->device_list() ||
      reduce_scatter->replica_groups().empty()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  const int64_t num_devices =
      reduce_scatter->device_list()->num_devices_per_group();
  if (!llvm::has_single_bit(static_cast<uint64_t>(num_devices))) {
    return absl::UnimplementedError(absl::StrFormat(
        "Collective kernels are only supported for power of 2 number of "
        "devices. Got %d.",
        num_devices));
  }
  const PrimitiveType element_type = reduce_scatter->shape().element_type();
  const std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(reduce_scatter->to_apply());
  if (!reduction_kind.has_value()) {
    return absl::UnimplementedError(
        "Unsupported reduction computation for the reduce-scatter kernel.");
  }
  // Support the same element type and reduction combinations as the one-shot
  // all-reduce kernel.
  if (!absl::c_linear_search(
          SupportedReductionOps(element_type),
          ReductionKindToOpcode(*reduction_kind, element_type))) {
    return absl::UnimplementedError(absl::StrFormat(
        "Element type (%s) and reduction kind (%v) combination is not "
        "supported for the reduce-scatter kernel.",
        primitive_util::LowercasePrimitiveTypeName(element_type),
        *reduction_kind));
  }
  if (!IsReduceScatterFlattenable(reduce_scatter)) {
    return absl::UnimplementedError(
        "Reduce-scatter kernel requires the scatter dimension to be the "
        "physically major-most non-trivial dimension and matching layouts.");
  }
  if (ShapeUtil::ElementsIn(reduce_scatter->shape()) % num_devices != 0) {
    return absl::UnimplementedError(
        "Reduce-scatter kernel requires the output element count to be "
        "divisible by the number of devices.");
  }
  const int64_t input_bytes =
      ShapeUtil::ByteSizeOf(reduce_scatter->operand(0)->shape());
  if (input_bytes > kMaxReduceScatterSizeBytes) {
    return absl::UnimplementedError(
        "Custom reduce-scatter strategy is only supported for small inputs.");
  }
  ABSL_ASSIGN_OR_RETURN(
      const bool is_local,
      IsAllReplicasLocal(gpu_topology, *reduce_scatter, device_assignment));
  if (!is_local) {
    return absl::UnimplementedError(
        "Cross-host symmetric memory collectives are not supported.");
  }
  return ReduceScatterInfo{
      /*.num_devices=*/num_devices,
      /*.num_output_elements=*/ShapeUtil::ElementsIn(reduce_scatter->shape()),
  };
}

LaunchDimensions ReduceScatterLaunchDimensions(
    int64_t num_output_elements, int64_t num_ranks,
    const se::DeviceDescription& device_info) {
  constexpr int64_t kNumElementsPerThread = 4;
  constexpr uint64_t kMaxThreadsPerBlock = 512;
  const int64_t warp_size = device_info.threads_per_warp();
  const int64_t elements_per_rank_tile =
      CeilOfRatio(num_output_elements, num_ranks);
  const int64_t threads_per_rank_tile = RoundUpTo(
      CeilOfRatio(elements_per_rank_tile, kNumElementsPerThread), warp_size);
  const int64_t threads_per_block = static_cast<int64_t>(
      std::min(kMaxThreadsPerBlock,
               llvm::bit_ceil(static_cast<uint64_t>(threads_per_rank_tile))));
  const int64_t blocks_per_rank_tile =
      CeilOfRatio(threads_per_rank_tile, threads_per_block);
  const int64_t blocks_per_grid =
      std::min(kReduceScatterMaxBlocksPerGrid,
               num_ranks * static_cast<int64_t>(llvm::bit_floor(
                               static_cast<uint64_t>(blocks_per_rank_tile))));
  return LaunchDimensions(blocks_per_grid, threads_per_block);
}

absl::StatusOr<CollectiveKernelSpec> CreateReduceScatterKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions) {
  int64_t group_size = instr->GetModule()->config().replica_count();
  if (!instr->replica_groups().empty() &&
      instr->replica_groups()[0].replica_ids_size() > 0) {
    group_size = instr->replica_groups()[0].replica_ids_size();
  }

  const int64_t input_size_bytes =
      ShapeUtil::ByteSizeOf(instr->operand(0)->shape());
  const int64_t num_signal_flags = group_size * launch_dimensions.num_blocks();
  const int64_t signal_size = xla::RoundUpTo<uint64_t>(
      num_signal_flags * sizeof(int32_t), kXlaAllocatedBufferAlignBytes);
  const int64_t remote_size =
      xla::RoundUpTo<uint64_t>(input_size_bytes, kXlaAllocatedBufferAlignBytes);

  const DebugOptions& debug_options =
      instr->GetModule()->config().debug_options();
  const SymmetricMemoryType sym_mem_type =
      IsCrossHostOneShotKernelEnabled(debug_options,
                                      DebugOptions::REDUCESCATTER)
          ? SymmetricMemoryType::kLoadStoreAccessible
          : SymmetricMemoryType::kXlaRendezvous;

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
           {KernelArgType::kScratchBuffer, /*index=*/0},  // signal buffers
           {KernelArgType::kScratchBuffer,
            /*index=*/1}},  // remote input buffer pointer table
          /* .sync_count_increment= */ 1u,
          /* .device_sync_count= */ true},
      /* .scratch_buffers= */
      {{signal_size, /*requires_multimem=*/false, sym_mem_type,
        /*should_memzero=*/true,
        /*should_double_buffer=*/true},
       {remote_size, /*requires_multimem=*/false, sym_mem_type,
        /*should_memzero=*/false,
        /*should_double_buffer=*/true}}};
  return kernel_spec;
}

}  // namespace xla::gpu
