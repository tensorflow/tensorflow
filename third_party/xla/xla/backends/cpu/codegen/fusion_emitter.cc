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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/alignment.h"
#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"
#include "xla/backends/cpu/codegen/symbol_name_util.h"
#include "xla/backends/cpu/codegen/tiled/tiled_fusion_emitter.h"
#include "xla/codegen/emitters/concatenate_kernel_emitter.h"
#include "xla/codegen/emitters/dynamic_update_slice_kernel_emitter.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/emitters/loop_kernel_emitter.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"
#include "xla/runtime/work_tile_size.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/backend_config.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

using ::mlir::MLIRContext;

static absl::StatusOr<std::string> GetName(const HloFusionInstruction& fusion,
                                           bool use_unique_c_name) {
  if (!use_unique_c_name) {
    return std::string(fusion.name());
  }

  return ConvertToCName(
      absl::StrCat(fusion.GetModule()->name(), "_", fusion.name()));
}

static HloInstructionAdaptor FindNonTrivialHero(
    const HloInstructionAdaptor& instr) {
  HloInstructionAdaptor hero = instr;

  // Go up the chain of trivial element-wise(+bitcast, -copy) operations. Note
  // that no memoization is needed due to number of operands constraints: we
  // never have to revisit same nodes.
  while (IsIntermediate(&hero.instruction(), /*allowed_operand_count=*/1) &&
         hero.parent().ContainsInstruction(hero.GetOperand(0))) {
    hero = hero.GetOperand(0);
  }

  // Try a bit harder to find a concat hero. The concat emitter also work if
  // there are elementwise ops with more than 1 operand on the path between root
  // and the root op.
  auto is_concatenate = [](const HloInstruction& node) {
    return node.opcode() == HloOpcode::kConcatenate;
  };
  if (auto concatenate = FindHero(hero, std::move(is_concatenate))) {
    return *concatenate;
  }

  // We just want to use the root for the loop emitter.
  return instr;
}

static const HloInstruction& FindNonTrivialHero(const HloInstruction& instr) {
  CHECK_NE(instr.opcode(), HloOpcode::kFusion);
  auto fusion_adaptor = HloFusionAdaptor::ForComputation(instr.parent());
  HloInstructionAdaptor instr_adaptor(instr, fusion_adaptor.get());
  return FindNonTrivialHero(instr_adaptor).instruction();
}

emitters::KernelArguments::BufferAlignment GetDefaultBufferAlignment() {
  emitters::KernelArguments::BufferAlignment buffer_alignment;
  buffer_alignment.entry_parameter_align_bytes = MinAlign();
  buffer_alignment.xla_allocated_buffer_align_bytes = MinAlign();
  buffer_alignment.constant_buffer_align_bytes = MinAlign();

  return buffer_alignment;
}

static int64_t GetWorkGroupCount(const HloFusionInstruction& fusion) {
  auto backend_config_or = fusion.backend_config<BackendConfig>();
  if (!backend_config_or.ok()) {
    return 1;
  }
  absl::Span<const int64_t> outer_dimension_partitions =
      backend_config_or->outer_dimension_partitions();

  if (outer_dimension_partitions.empty()) {
    return 1;
  }

  return absl::c_accumulate(outer_dimension_partitions, 1,
                            std::multiplies<int64_t>());
}

WorkDimensions GetWorkDimensions(const Shape& shape,
                                 const HloFusionInstruction& fusion) {
  auto minor_to_major = LayoutUtil::MinorToMajor(shape.layout());

  if (minor_to_major.empty()) {
    return WorkDimensions{};
  }

  int64_t work_group_count = GetWorkGroupCount(fusion);
  NumWorkGroups num_work_groups{static_cast<uint64_t>(work_group_count)};

  WorkTileSize work_tile_size;
  int64_t folded_dims = 1;
  for (int64_t dim : llvm::reverse(minor_to_major)) {
    int64_t dim_size = ShapeUtil::GetDimension(shape, dim);
    int64_t accumilated_dim_size = folded_dims * dim_size;
    if (accumilated_dim_size < work_group_count) {
      folded_dims *= dim_size;
    } else if (work_group_count != 1) {
      work_tile_size.dimensions.push_back(
          CeilOfRatio(accumilated_dim_size, work_group_count));
      work_group_count = 1;
    } else {
      work_tile_size.dimensions.push_back(dim_size);
    }
  }

  return WorkDimensions{NumWorkClusters{}, num_work_groups, NumWorkItems{},
                        work_tile_size};
}

// Get the work dimensions for the given fusion.
// TODO(willfroom): Make this have more smarts to give better dims.
static WorkDimensions GetLoopEmitterWorkDims(const HloFusionInstruction& fusion,
                                             const HloFusionSpec& fusion_spec) {
  Shape indexing_shape =
      emitters::LoopFusionKernelEmitter::GetIndexingShape(fusion_spec);

  return GetWorkDimensions(indexing_shape, fusion);
}

static WorkDimensions GetConcatenateEmitterWorkDims(
    const HloFusionInstruction& fusion, const HloFusionSpec& fusion_spec) {
  Shape indexing_shape =
      emitters::ConcatenateFusionKernelEmitter::GetIndexingShape(fusion_spec);

  return GetWorkDimensions(indexing_shape, fusion);
}

static WorkDimensions GetDynamicUpdateSliceEmitterWorkDims(
    const HloFusionInstruction& fusion, const HloFusionSpec& fusion_spec) {
  Shape indexing_shape =
      emitters::DynamicUpdateSliceKernelEmitter::GetIndexingShape(fusion_spec);

  return GetWorkDimensions(indexing_shape, fusion);
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
  absl::InlinedVector<HloInstructionAdaptor, 2> heroes = {
      FindNonTrivialHero(fusion_adaptor->GetRoots().front())};

  return HloFusionSpec(std::move(fusion_adaptor), std::move(roots),
                       std::move(heroes));
}

static absl::StatusOr<KernelDefinition<MlirKernelSource>> EmitLoopFusionKernel(
    MLIRContext& context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, absl::string_view name) {
  VLOG(2) << "Emitting loop fusion kernel: " << name;
  HloFusionSpec fusion_spec = GetLoopFusionSpec(fusion);
  auto work_dimensions = GetLoopEmitterWorkDims(fusion, fusion_spec);

  emitters::LoopFusionKernelEmitter loop_fusion_emitter(
      context, fusion, std::move(fusion_spec), buffer_assignment,
      GetDefaultBufferAlignment(), work_dimensions, name, BackendKind::kCpu);
  TF_ASSIGN_OR_RETURN(auto mlir_kernel_definition,
                      loop_fusion_emitter.EmitKernelDefinition());

  mlir::OpBuilder builder(&context);
  mlir_kernel_definition.source().module().getOperation()->setAttr(
      xla::CpuMemoryRegionNameAttr::name,
      builder.getStringAttr(
          BuildModuleMemoryRegionName(loop_fusion_emitter.name(), &fusion)));

  return mlir_kernel_definition;
}

static absl::StatusOr<KernelDefinition<MlirKernelSource>>
EmitConcatenateFusionKernel(MLIRContext& context,
                            const HloFusionInstruction& fusion,
                            const BufferAssignment* buffer_assignment,
                            absl::string_view name) {
  VLOG(2) << "Emitting concatenate fusion kernel: " << name;
  HloFusionSpec fusion_spec = GetLoopFusionSpec(fusion);
  auto work_dimensions = GetConcatenateEmitterWorkDims(fusion, fusion_spec);

  emitters::ConcatenateFusionKernelEmitter concatenate_fusion_emitter(
      context, fusion, std::move(fusion_spec), buffer_assignment,
      GetDefaultBufferAlignment(), work_dimensions, name, BackendKind::kCpu);
  TF_ASSIGN_OR_RETURN(auto mlir_kernel_definition,
                      concatenate_fusion_emitter.EmitKernelDefinition());

  mlir::OpBuilder builder(&context);
  mlir_kernel_definition.source().module().getOperation()->setAttr(
      xla::CpuMemoryRegionNameAttr::name,
      builder.getStringAttr(BuildModuleMemoryRegionName(
          concatenate_fusion_emitter.name(), &fusion)));

  return mlir_kernel_definition;
}

static absl::StatusOr<KernelDefinition<MlirKernelSource>>
EmitDynamicUpdateSliceFusionKernel(MLIRContext& context,
                                   const HloFusionInstruction& fusion,
                                   const BufferAssignment* buffer_assignment,
                                   absl::string_view name) {
  VLOG(2) << "Emitting dynamic update slice fusion kernel: " << name;
  HloFusionSpec fusion_spec = GetLoopFusionSpec(fusion);
  auto work_dimensions =
      GetDynamicUpdateSliceEmitterWorkDims(fusion, fusion_spec);

  emitters::DynamicUpdateSliceKernelEmitter emitter(
      context, fusion, std::move(fusion_spec), buffer_assignment,
      GetDefaultBufferAlignment(), work_dimensions, name, BackendKind::kCpu);
  TF_ASSIGN_OR_RETURN(auto mlir_kernel_definition,
                      emitter.EmitKernelDefinition());

  mlir::OpBuilder builder(&context);
  mlir_kernel_definition.source().module().getOperation()->setAttr(
      xla::CpuMemoryRegionNameAttr::name,
      builder.getStringAttr(
          BuildModuleMemoryRegionName(emitter.name(), &fusion)));

  return mlir_kernel_definition;
}

absl::StatusOr<KernelDefinition<MlirKernelSource>> EmitFusionKernel(
    MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const BufferAssignment* buffer_assignment, bool use_unique_c_name,
    bool enable_tiled_emitter) {
  TF_ASSIGN_OR_RETURN(std::string name, GetName(fusion, use_unique_c_name));

  if (enable_tiled_emitter) {
    if (auto tiling_or = GetTilingIfSupported(mlir_context, fusion);
        tiling_or.ok()) {
      return EmitTiledFusionKernel(mlir_context, fusion, buffer_assignment,
                                   name, GetWorkGroupCount(fusion),
                                   std::move(*tiling_or));
    }
  }

  if (fusion.fusion_kind() == HloFusionInstruction::FusionKind::kLoop) {
    const HloInstruction& hero =
        FindNonTrivialHero(*fusion.fused_expression_root());
    if (hero.opcode() == HloOpcode::kConcatenate) {
      return EmitConcatenateFusionKernel(mlir_context, fusion,
                                         buffer_assignment, name);
    }
    auto fusion_spec = GetLoopFusionSpec(fusion);
    if (IsDynamicUpdateSliceFusion(fusion_spec)) {
      TF_ASSIGN_OR_RETURN(
          bool dus_inplace,
          CanEmitFusedDynamicUpdateSliceInPlace(fusion_spec.fusion(),
                                                buffer_assignment, &fusion));
      if (dus_inplace) {
        return EmitDynamicUpdateSliceFusionKernel(mlir_context, fusion,
                                                  buffer_assignment, name);
      }
    }
    return EmitLoopFusionKernel(mlir_context, fusion, buffer_assignment, name);
  }

  return absl::UnimplementedError("Fusion kind not supported.");
}

}  // namespace xla::cpu
