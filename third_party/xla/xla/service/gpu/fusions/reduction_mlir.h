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

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/computation_partitioner.h"
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"
#include "xla/service/gpu/fusions/reduction_base.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/reduction_utils.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

using HloValueMap =
    absl::flat_hash_map<const HloInstruction*, llvm::SmallVector<mlir::Value>>;

// Reduction fusion. Lowers to LLVM via MLIR. Currently not fully
// implemented: only single reduction groups, no side outputs, only row
// reductions.
class MlirReductionFusion : public MlirFusionEmitterBase {
 public:
  explicit MlirReductionFusion(const HloFusionAnalysis& analysis);

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override = 0;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override;

  LaunchDimensions launch_dimensions() const override;

  const ReductionGroups& GetGroups() const { return groups_; }

 protected:
  struct EmitterState;
  friend struct EmitterState;

  absl::Status EmitEntryFunction(
      const mlir_converter::PartitionedComputations& computations,
      const mlir_converter::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  std::vector<mlir_converter::EpilogueSpecification> GetEpilogues(
      const HloFusionInstruction& fusion,
      mlir::MLIRContext* mlir_context) const override;

  llvm::SmallVector<mlir::Value> EvaluateEpilogue(
      mlir::ImplicitLocOpBuilder& b, const HloValueMap& results,
      llvm::SmallVector<mlir::Value> outputs, EmitterState& state, int group_id,
      mlir::MLIRContext* ctx, mlir::Value vector_index = nullptr) const;

  virtual llvm::SmallVector<mlir::Value> EmitReduction(
      int group_id, EmitterState& state) const = 0;

  Shape GetReduceOperandShape() const {
    return first_reduce_->operand(0)->shape();
  }

  virtual IndexingMap ComputeThreadIdToReductionInputIndexing(
      mlir::MLIRContext* ctx) const = 0;

  // The reduction heroes for each reduction group.
  std::vector<std::vector<const HloInstruction*>> reduction_heroes_;
  // The roots that have reduction heroes for each reduction group.
  std::vector<std::vector<const HloInstruction*>> reduction_roots_;
  // The side output roots for each reduction group.
  std::vector<std::vector<const HloInstruction*>> side_output_roots_;
  const HloFusionAnalysis& analysis_;

  // The number of elements in each dimension.
  absl::InlinedVector<int64_t, 4> input_shape_;

  // The number of elements for each dimension of a tile.
  absl::InlinedVector<int64_t, 4> tile_sizes_per_thread_;
  absl::InlinedVector<int64_t, 4> tile_sizes_per_block_;

  absl::InlinedVector<int64_t, 4> num_threads_;
  absl::InlinedVector<int64_t, 4> num_blocks_;
  int64_t total_num_blocks_;
  int64_t total_num_threads_per_block_;
  int64_t vector_size_ = -1;

  ReductionDimensions reduction_dimensions_;
  ReductionGroups groups_;
  const HloInstruction* first_reduce_;
};

class MlirRowReductionFusion : public MlirReductionFusion {
 public:
  explicit MlirRowReductionFusion(const HloFusionAnalysis& analysis);

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override;

 protected:
  int GetRowsPerWarp() const;
  llvm::SmallVector<mlir::Value> EmitReduction(
      int group_id, EmitterState& state) const override;
  IndexingMap ComputeThreadIdToReductionInputIndexing(
      mlir::MLIRContext* ctx) const override;
};

class MlirColumnReductionFusion : public MlirReductionFusion {
 public:
  explicit MlirColumnReductionFusion(const HloFusionAnalysis& analysis);

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override;

 protected:
  llvm::SmallVector<mlir::Value> EmitReduction(
      int group_id, EmitterState& state) const override;
  IndexingMap ComputeThreadIdToReductionInputIndexing(
      mlir::MLIRContext* ctx) const override;
  int64_t num_warps_per_column_;
  int64_t num_blocks_per_row_;
};

std::unique_ptr<MlirReductionFusion> CreateMlirReductionFusion(
    const HloFusionAnalysis& analysis);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_REDUCTION_MLIR_H_
