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
#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/gpu/codegen/emitters/emitter_base.h"
#include "xla/codegen/emitters/computation_partitioner.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/launch_dimensions.h"

namespace xla {
namespace gpu {

// Fusion node where the root is either:
// 1. a dynamic-update-slice op
// 2. a bitcast of a dynamic-update-slice op
// 3. a tuple op returning the result of several dynamic-update-slice ops
// 4. a tuple op returning the result of several bitcast
//    dynamic-update-slice ops
//
// Lowers to LLVM via MLIR.
class InPlaceDynamicUpdateSliceFusion : public EmitterBase {
 public:
  explicit InPlaceDynamicUpdateSliceFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis),
        dus_ops_(GetOutputDefiningDynamicUpdateSlices(analysis.fusion_roots())),
        config_(ComputeLoopFusionConfig(
            analysis, dus_ops_[0].instruction().operand(1)->shape())) {}

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* indexing_context) const override {
    // The mapping cannot be statically computed in general, since the offsets
    // are unknown.
    return std::nullopt;
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* indexing_context) const override;

 protected:
  absl::Status EmitEntryFunction(
      const emitters::PartitionedComputations& computations,
      const emitters::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  std::vector<emitters::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

 private:
  const HloFusionAnalysis& analysis_;
  std::vector<HloInstructionAdaptor> dus_ops_;
  LaunchDimensionsConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_
