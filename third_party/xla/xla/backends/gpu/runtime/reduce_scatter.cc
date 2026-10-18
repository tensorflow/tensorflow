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

#include <cstdint>
#include <optional>
#include <vector>

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

// A broader check for reduce-scatter kernel support that verifies device,
// operand count, replica group, reduction computation, element-type, layout,
// and size constraints.
// Returns absl::OkStatus() if supported, or an absl::UnimplementedError
// explaining why not.
absl::Status IsReduceScatterKernelSupported(
    const se::DeviceDescription& device_info,        //
    int32_t num_operands,                            //
    std::optional<ReductionKind> reduction_kind,     //
    int64_t num_devices,                             //
    int64_t num_output_elements,                     //
    PrimitiveType element_type,                      //
    bool is_flattenable,                             //
    bool is_local,                                   //
    const std::vector<ReplicaGroup>& replica_groups  //
) {
  if (!device_info.cuda_compute_capability().IsAtLeastHopper()) {
    return absl::UnimplementedError(absl::StrFormat(
        "Triton reduce-scatter requires CUDA compute capability >= 9.0. "
        "Got: %s.",
        device_info.gpu_compute_capability().ToString()));
  }
  // TODO(b/383125489): Support variadic arguments.
  if (num_operands != 1) {
    return absl::UnimplementedError(
        "Reduce-scatter kernel only supports a single operand.");
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
  if (!is_flattenable) {
    return absl::UnimplementedError(
        "Reduce-scatter kernel requires the scatter dimension to be the "
        "physically major-most non-trivial dimension and matching layouts.");
  }
  if (num_output_elements % num_devices != 0) {
    return absl::UnimplementedError(
        "Reduce-scatter kernel requires the output element count to be "
        "divisible by the number of devices.");
  }
  const int64_t input_bytes = num_output_elements * num_devices *
                              primitive_util::ByteWidth(element_type);
  if (input_bytes > kMaxReduceScatterSizeBytes) {
    return absl::UnimplementedError(
        "Custom reduce-scatter strategy is only supported for small inputs.");
  }
  return absl::OkStatus();
}

absl::StatusOr<ReduceScatterInfo> BuildReduceScatterInfo(
    const GpuTopology& gpu_topology,
    const HloReduceScatterInstruction* reduce_scatter,
    const DeviceAssignment* device_assignment) {
  if (!gpu_topology.has_gpu_target_config()) {
    return absl::InvalidArgumentError(
        "GpuTopology must have a target config to build ReduceScatterInfo.");
  }
  const se::DeviceDescription& device_info =
      gpu_topology.gpu_target_config().device_description;
  if (!reduce_scatter->device_list()) {
    return absl::UnimplementedError(
        "Replica groups must be explicitly provided for collective kernels.");
  }
  const int64_t num_devices =
      reduce_scatter->device_list()->num_devices_per_group();
  const int32_t num_operands = reduce_scatter->operand_count();
  const std::optional<ReductionKind> reduction_kind =
      MatchReductionComputation(reduce_scatter->to_apply());
  const PrimitiveType element_type =
      reduce_scatter->operand(0)->shape().element_type();
  const int64_t num_output_elements =
      reduce_scatter->shape().IsArray()
          ? ShapeUtil::ElementsIn(reduce_scatter->shape())
          : 0;
  const bool is_flattenable = IsReduceScatterFlattenable(reduce_scatter);
  if (device_info.device_interconnect_info().active_links <= 0) {
    return absl::UnimplementedError(
        "Collective kernels are only supported on devices with NVLink "
        "support.");
  }
  ABSL_ASSIGN_OR_RETURN(const bool is_local,
                   AreAllReplicasOnSameSlice(gpu_topology, *reduce_scatter,
                                             device_assignment));
  ABSL_RETURN_IF_ERROR(IsReduceScatterKernelSupported(
      device_info, num_operands, reduction_kind, num_devices,
      num_output_elements, element_type, is_flattenable, is_local,
      reduce_scatter->replica_groups()));
  return ReduceScatterInfo{
      /*.num_devices=*/num_devices,
      /*.num_output_elements=*/num_output_elements,
  };
}

absl::StatusOr<CollectiveKernelSpec> CreateReduceScatterKernelSpec(
    const HloInstruction* instr, const LaunchDimensions& launch_dimensions,
    SymmetricMemoryType scratch_memory_type) {
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
      {{signal_size, /*requires_multimem=*/false, scratch_memory_type,
        /*should_memzero=*/true,
        /*should_double_buffer=*/true},
       {remote_size, /*requires_multimem=*/false, scratch_memory_type,
        /*should_memzero=*/false,
        /*should_double_buffer=*/true}}};
  return kernel_spec;
}

}  // namespace xla::gpu
