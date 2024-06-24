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
#ifndef XLA_SERVICE_GPU_FUSIONS_REDUCTION_MLIR_H_
#define XLA_SERVICE_GPU_FUSIONS_REDUCTION_MLIR_H_

#include <vector>

#include "absl/status/status.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"
#include "xla/service/gpu/fusions/reduction_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla {
namespace gpu {

class MlirReductionInfo {
 public:
  static MlirReductionInfo Create(const HloFusionAnalysis& analysis);

  const Tiling& GetTiling() const { return tiling_; }
  const ReductionGroups& GetGroups() const { return groups_; }
  Shape GetReduceOperandShape() const {
    return first_reduce_->operand(0)->shape();
  }

  bool IsRowReduction() const { return is_row_reduction_; }
  bool IsRaceFree() const { return is_race_free_; }
  int GetRowsPerWarp() const;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const;

  LaunchDimensions launch_dimensions() const;

 private:
  MlirReductionInfo(const HloFusionAnalysis& analysis, Tiling tiling,
                    bool is_row_reduction, bool is_race_free,
                    ReductionGroups groups, const HloInstruction* first_reduce)
      : analysis_(analysis),
        tiling_(tiling),
        is_row_reduction_(is_row_reduction),
        is_race_free_(is_race_free),
        groups_(std::move(groups)),
        first_reduce_(first_reduce) {}

  void AddGroupIdConstraint(IndexingMap& map, int64_t root_index,
                            mlir::MLIRContext* ctx) const;

  const HloFusionAnalysis& analysis_;
  Tiling tiling_;
  bool is_row_reduction_;
  bool is_race_free_;
  ReductionGroups groups_;
  const HloInstruction* first_reduce_;
};

// Reduction fusion. Lowers to LLVM via MLIR. Currently not fully
// implemented: only single reduction groups, no side outputs, only row
// reductions.
class MlirReductionFusion : public MlirFusionEmitterBase {
 public:
  explicit MlirReductionFusion(const HloFusionAnalysis& analysis);

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    return reduction_info_.ComputeThreadIdToOutputIndexing(root_index, ctx);
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override {
    return reduction_info_.ComputeThreadIdToInputIndexing(
        root_index, hero_operand_index, ctx);
  }

  LaunchDimensions launch_dimensions() const override {
    return reduction_info_.launch_dimensions();
  }

  const MlirReductionInfo& reduction_info() const { return reduction_info_; }

 protected:
  absl::Status EmitEntryFunction(
      const mlir_converter::PartitionedComputations& computations,
      const mlir_converter::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  std::vector<mlir_converter::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

 private:
  struct EmitterState;
  friend struct EmitterState;

  llvm::SmallVector<mlir::Value> EmitReduction(int group_id,
                                               EmitterState& state) const;

  // The reduction heroes for each reduction group.
  std::vector<std::vector<const HloInstruction*>> reduction_heroes_;
  // The roots that have reduction heroes for each reduction group.
  std::vector<std::vector<const HloInstruction*>> reduction_roots_;
  // The side output roots for each reduction group.
  std::vector<std::vector<const HloInstruction*>> side_output_roots_;
  const HloFusionAnalysis& analysis_;
  MlirReductionInfo reduction_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_REDUCTION_MLIR_H_
