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

#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_CONCATENATE_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_CONCATENATE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

class ConcatenateFusion final : public EmitterBase {
 public:
  explicit ConcatenateFusion(const HloFusionAnalysis& analysis);

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override;

  std::optional<std::vector<IndexingMap>> ComputeThreadIdToInputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override;

 protected:
  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> CreateMLIRModule(
      mlir::MLIRContext& mlir_context, const HloFusionInstruction& fusion,
      const std::string& entry_function_name,
      const BufferAssignment* buffer_assignment) const override;

  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  WorkDimensions GetWorkDimensions() const;

 private:
  const HloFusionAnalysis& analysis_;
  Shape largest_shape_;
  LaunchDimensionsConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_CONCATENATE_H_
