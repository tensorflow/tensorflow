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

#include "xla/backends/cpu/codegen/fusion_emitter.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/loop_kernel_emitter.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/mlir_kernel_definition.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla::cpu {

static emitters::KernelArguments::BufferAlignment GetDefaultBufferAlignment() {
  emitters::KernelArguments::BufferAlignment buffer_alignment;
  buffer_alignment.entry_parameter_align_bytes = MinAlign();
  buffer_alignment.xla_allocated_buffer_align_bytes = Align();
  buffer_alignment.constant_buffer_align_bytes = Align();

  return buffer_alignment;
}

static constexpr uint64_t kMaxWorkItems = 1024;
static constexpr uint64_t kUnrollFactor = 8;

static WorkDimensions GetDefaultWorkDimensions(uint64_t group_count,
                                               uint64_t total_item_count) {
  NumWorkGroups num_work_groups{group_count};
  NumWorkItems num_work_items{std::min<uint64_t>(
      kMaxWorkItems,
      CeilOfRatio(total_item_count, group_count * kUnrollFactor))};

  return WorkDimensions{NumWorkClusters(), num_work_groups, num_work_items};
}

// Get the work dimensions for the given fusion.
// TODO(willfroom): Make this have more smarts to give better dims.
static WorkDimensions GetLoopEmitterWorkDims(const HloFusionInstruction& fusion,
                                             const HloFusionSpec& fusion_spec) {
  Shape indexing_shape =
      emitters::LoopFusionKernelEmitter::GetIndexingShape(fusion_spec);
  auto total_item_count = ShapeUtil::ElementsIn(indexing_shape);

  auto backend_config_or = fusion.backend_config<BackendConfig>();
  if (!backend_config_or.ok()) {
    return GetDefaultWorkDimensions(1, total_item_count);
  }
  absl::Span<const int64_t> outer_dimension_partitions =
      backend_config_or->outer_dimension_partitions();

  if (outer_dimension_partitions.empty()) {
    return GetDefaultWorkDimensions(1, total_item_count);
  }

  int64_t num_groups = absl::c_accumulate(outer_dimension_partitions, 1,
                                          std::multiplies<int64_t>());

  return GetDefaultWorkDimensions(num_groups, total_item_count);
}

static HloFusionSpec GetLoopFusionSpec(const HloFusionInstruction& fusion) {
  // Crash OK, this is checked in the caller.
  CHECK(fusion.fusion_kind() == HloFusionInstruction::FusionKind::kLoop);

  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForInstruction(&fusion);
  // Loop fusion simply uses the roots as the hero instructions.
  // When we want to support more complex fusion kinds, we will need to port the
  // GPU code to get the non-trivial hero instructions.
  absl::InlinedVector<HloInstructionAdaptor, 2> roots =
      fusion_adaptor->GetRoots();
  absl::InlinedVector<HloInstructionAdaptor, 2> heroes =
      fusion_adaptor->GetRoots();

  return HloFusionSpec(std::move(fusion_adaptor), std::move(roots),
                       std::move(heroes));
}

absl::StatusOr<MlirKernelDefinition> EmitFusionKernel(
    mlir::MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment) {
  if (fusion.fusion_kind() == HloFusionInstruction::FusionKind::kLoop) {
    VLOG(2) << "Emitting loop fusion kernel: " << fusion.name();
    HloFusionSpec fusion_spec = GetLoopFusionSpec(fusion);
    WorkDimensions work_dimensions =
        GetLoopEmitterWorkDims(fusion, fusion_spec);
    return emitters::LoopFusionKernelEmitter(
               context, fusion, std::move(fusion_spec), buffer_assignment,
               GetDefaultBufferAlignment(), work_dimensions, kUnrollFactor,
               fusion.name(), BackendKind::kCpu)
        .EmitKernelDefinition();
  }

  return absl::UnimplementedError("Fusion kind not supported.");
}

}  // namespace xla::cpu
