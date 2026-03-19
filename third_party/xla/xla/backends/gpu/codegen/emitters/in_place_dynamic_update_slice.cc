/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/emitters/in_place_dynamic_update_slice.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/dynamic_update_slice_kernel_emitter.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::MLIRContext;

constexpr int kDUSUpdateIndex = 1;

}  // namespace

LaunchDimensions InPlaceDynamicUpdateSliceFusion::launch_dimensions() const {
  const auto& update_shape =
      dus_ops_.front().GetOperand(kDUSUpdateIndex).shape();
  return CalculateLaunchDimensions(update_shape, analysis_.device_info(),
                                   config_);
}

std::optional<std::vector<IndexingMap>>
InPlaceDynamicUpdateSliceFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, MLIRContext* mlir_context) const {
  // TODO(b/331355203): Implement thread ID -> operand indexing.
  std::vector<IndexingMap> result(
      analysis_.fusion_hero(root_index).GetOperands().size(),
      IndexingMap::GetUndefined());

  using KernelEmitter = emitters::DynamicUpdateSliceKernelEmitter;
  result[kDUSUpdateIndex] = KernelEmitter::ComputeWorkItemIdToOutputIndexing(
      GetWorkDimensions(),
      KernelEmitter::GetIndexingShape(analysis_.fusion_spec()), mlir_context);
  return result;
}

std::vector<emitters::EpilogueSpecification>
InPlaceDynamicUpdateSliceFusion::GetEpilogues(
    const HloFusionInstruction& fusion, MLIRContext* mlir_context) const {
  // We don't actually support epilogues for DUS, but this is how we tell
  // the base class that we don't want it to generate code for the DUS.
  std::vector<emitters::EpilogueSpecification> epilogues;
  for (const auto& [dus_op, root] :
       llvm::zip(dus_ops_, analysis_.fusion_roots())) {
    epilogues.push_back(emitters::EpilogueSpecification::FromIdentityIndexing(
        &dus_op.instruction(), &root.instruction(), mlir_context));
  }
  return epilogues;
}

WorkDimensions InPlaceDynamicUpdateSliceFusion::GetWorkDimensions() const {
  WorkDimensions work_dimensions = launch_dimensions().AsWorkDimensions();
  work_dimensions.work_tile_size.dimensions.push_back(config_.unroll_factor);
  return work_dimensions;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
InPlaceDynamicUpdateSliceFusion::CreateMLIRModule(
    mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
    const std::string& entry_function_name,
    const BufferAssignment* buffer_assignment) const {
  emitters::DynamicUpdateSliceKernelEmitter emitter(
      mlir_context, fusion, analysis_.fusion_spec(), buffer_assignment,
      GetDefaultBufferAlignment(), GetWorkDimensions(), entry_function_name,
      BackendKind::kGpu);

  TF_ASSIGN_OR_RETURN(auto kernel_definition, emitter.EmitKernelDefinition());
  return std::move(kernel_definition).TakeSource().TakeModule();
}

absl::Status InPlaceDynamicUpdateSliceFusion::EmitEntryFunction(
    const emitters::PartitionedComputations& computations,
    const emitters::CallTargetProvider& call_targets,
    mlir::func::FuncOp entry_function,
    const HloFusionInstruction& fusion) const {
  return absl::UnimplementedError("Not implemented");
}

}  // namespace gpu
}  // namespace xla
