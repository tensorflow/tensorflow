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
#ifndef XLA_SERVICE_GPU_FUSIONS_REDUCTION_BASE_H_
#define XLA_SERVICE_GPU_FUSIONS_REDUCTION_BASE_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/fusions/tiling_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

struct ReductionGroups {
  std::vector<std::vector<const HloInstruction*>> grouped_roots;

  // For each root of the fusion, returns the index of the group it was placed
  // in.
  std::vector<int> group_id_per_root;

  // For each root of the fusion, returns whether it is a reduction root, or
  // an additional output.
  std::vector<bool> is_reduction_root;
};

class ReductionInfo {
 public:
  static ReductionInfo Create(const HloFusionAnalysis& analysis, bool for_mlir);

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
  ReductionInfo(const HloFusionAnalysis& analysis, Tiling tiling,
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

// Base class for reduction fusions. Computes shared information (reduction
// grouping) and provides implementations of thread->input/output indexing.
template <typename Base, bool is_mlir = false>
class ReductionFusionBase : public Base {
 public:
  explicit ReductionFusionBase(const HloFusionAnalysis& analysis)
      : analysis_(analysis),
        reduction_info_(ReductionInfo::Create(analysis, is_mlir)) {}

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    return reduction_info().ComputeThreadIdToOutputIndexing(root_index, ctx);
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override {
    return reduction_info().ComputeThreadIdToInputIndexing(
        root_index, hero_operand_index, ctx);
  }

  LaunchDimensions launch_dimensions() const override {
    return reduction_info().launch_dimensions();
  }

  const ReductionInfo& reduction_info() const { return reduction_info_; }

  const HloFusionAnalysis& analysis() const { return analysis_; }

 private:
  const HloFusionAnalysis& analysis_;
  ReductionInfo reduction_info_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_REDUCTION_BASE_H_
