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

#ifndef XLA_CODEGEN_EMITTERS_DYNAMIC_UPDATE_SLICE_KERNEL_EMITTER_H_
#define XLA_CODEGEN_EMITTERS_DYNAMIC_UPDATE_SLICE_KERNEL_EMITTER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/codegen/emitters/ir/xla_ops.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/codegen/hlo_fusion_spec.h"
#include "xla/codegen/kernel_emitter.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/runtime/work_dimensions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"

namespace xla::emitters {

// Fusion node where the root is either:
// 1. a dynamic-update-slice op
// 2. a bitcast of a dynamic-update-slice op
// 3. a tuple op returning the result of several dynamic-update-slice ops
// 4. a tuple op returning the result of several bitcast
//    dynamic-update-slice ops
class DynamicUpdateSliceKernelEmitter final
    : public KernelEmitter<MlirKernelSource> {
 public:
  DynamicUpdateSliceKernelEmitter(
      mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
      const HloFusionSpec& fusion_spec,
      const BufferAssignment* buffer_assignment,
      KernelArguments::BufferAlignment buffer_alignment,
      WorkDimensions work_dimensions, absl::string_view entry_function_name,
      BackendKind backend_kind);

  absl::string_view name() const final {
    return "dynamic_update_slice_kernel_emitter";
  }

  absl::StatusOr<KernelDefinition> EmitKernelDefinition() override;

  // Get the shape that will be used for loop indexing for the given fusion
  // specification.
  static Shape GetIndexingShape(const HloFusionSpec& fusion_spec);
  // Get the mapping from work item id to output.
  static IndexingMap ComputeWorkItemIdToOutputIndexing(
      const WorkDimensions& work_dimensions, const Shape& update_shape,
      mlir::MLIRContext* ctx);

 private:
  IndexingMap ComputeWorkItemIdToInputIndexing(
      mlir::MLIRContext* mlir_context) const;
  absl::StatusOr<KernelSpec> GetKernelSpec() const;

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const;

  std::vector<emitters::EpilogueSpecification> GetEpilogues() const;

 private:
  mlir::MLIRContext& mlir_context_;
  const HloFusionInstruction& fusion_;
  const HloFusionSpec& fusion_spec_;
  std::vector<HloInstructionAdaptor> dus_ops_;
  const BufferAssignment* buffer_assignment_;
  KernelArguments::BufferAlignment buffer_alignment_;
  WorkDimensions work_dimensions_;
  std::string entry_function_name_;
  BackendKind backend_kind_;
};

}  // namespace xla::emitters

#endif  // XLA_CODEGEN_EMITTERS_DYNAMIC_UPDATE_SLICE_KERNEL_EMITTER_H_
