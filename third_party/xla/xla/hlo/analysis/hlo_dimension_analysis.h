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

#ifndef XLA_HLO_ANALYSIS_HLO_DIMENSION_ANALYSIS_H_
#define XLA_HLO_ANALYSIS_HLO_DIMENSION_ANALYSIS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"

namespace xla {

enum DimensionInfo : uint8_t {
  // kDotDependent indicates there is a DOT that can reach an operand of the
  // instruction. We want to use this information to distinguish between
  // WeightGradient and ActivationGradient as follows to help decide whether
  // we can overlap the all-gather/reduce-scatter with other dot operations
  // outside the chain:
  //
  // ActivationGradient: a DOT, there is another DOT that can reach the operands
  // of this DOT via def-use chain.
  //
  // WeightGradient: a DOT, no other DOT can reach the operand of this DOT via
  // def-use chain.
  //
  // Because we don't schedule instructions across computation boundaries, we
  // don't propagate kDotDependent across computation boundaries. On the other
  // hand, we propagate kWeight across computation boundaries.
  kWeight,
  kDotDependent,
  kUnknown,
};

inline std::string DimensionInfoToString(DimensionInfo dim_info) {
  switch (dim_info) {
    case DimensionInfo::kWeight:
      return "weight";
    case DimensionInfo::kDotDependent:
      return "dot_dependent";
    case DimensionInfo::kUnknown:
      return "unknown";
  }
}

using DimensionInfoMap =
    absl::node_hash_map<const HloInstruction*, ShapeTree<DimensionInfo>>;

// This analysis pass determines which HLO instructions produce/are weights.
// Parameters to the entry computation are considered weights, and this property
// is propagated through instructions that preserve it (slice, convert,
// etc).
class HloDimensionAnalysis {
 public:
  friend class HloDimensionInfoPropagation;
  static absl::StatusOr<std::unique_ptr<HloDimensionAnalysis>> Run(
      const HloModule& module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Whether the instruction has been annotated with dimension info.
  bool HasDimensionInfo(const HloInstruction* instruction) const {
    return info_map_.contains(instruction);
  }

  // Whether any leaf in the instruction shape is a weight.
  bool IsWeight(const HloInstruction* instruction) const;
  // Whether any leaf in the instruction shape is dot dependent.
  bool IsDotDependent(const HloInstruction* instruction) const;
  // Whether any leaf in the instructon shape is a weight or dot dependent.
  bool IsKnownDimensionInfo(const HloInstruction* instruction) const;

  // Returns map of HLO instructions to their dimension info.
  // If an instruction is not found in the map, it means that we have not
  // determined its dimension info.
  const DimensionInfoMap& GetDimensionInfoMap() const { return info_map_; }

  // Returns the dimension info for the given instruction.
  std::optional<ShapeTree<DimensionInfo>> GetDimensionInfo(
      const HloInstruction* instruction) const;

  bool IsDotOrHasDotDependent(const HloInstruction* op) const;

 protected:
  explicit HloDimensionAnalysis(
      const HloModule& module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      : module_(module), execution_threads_(execution_threads) {}

  // Sets the instruction DimensionInfo to indicate it is a weight or
  // dot-dependent. This is used to annotate the entry computation parameters
  // and other instructions that are known to be weights or dot-dependents.
  absl::Status SetDimensionInfo(const HloInstruction* instruction,
                                DimensionInfo value);

  // Sets the dimension info for the given target instruction.
  absl::Status SetDimensionInfo(const HloInstruction* target,
                                ShapeTree<DimensionInfo> annotation);

  // Annotates the entry computation parameters as weights.
  absl::Status AnnotateEntryComputationParameters(const HloModule& module);

  // Runs the analysis on the given computation to determine the DimensionInfo
  // for each instruction.
  absl::Status RunOnComputation(const HloComputation& computation);

  // Runs the analysis on the given computation, with the given operands as the
  // computation parameters. Propagates the dimension info from the callsite
  // operands to the computation parameters.
  absl::Status RunOnComputation(
      const HloComputation& computation,
      absl::Span<const HloInstruction* const> operands);

  DimensionInfoMap info_map_;
  const HloModule& module_;
  const absl::flat_hash_set<absl::string_view>& execution_threads_;
};

class HloDimensionInfoPropagation : public DfsHloVisitorWithDefault {
 public:
  explicit HloDimensionInfoPropagation(HloDimensionAnalysis* dimension_analysis)
      : analysis_(dimension_analysis) {}
  absl::Status Run(const HloComputation& computation);
  absl::Status DefaultAction(HloInstruction* instruction) override;
  // go/keep-sorted start
  absl::Status HandleAllGather(HloInstruction* all_gather) override;
  absl::Status HandleBitcast(HloInstruction* bitcast) override;
  absl::Status HandleBitcastConvert(HloInstruction* bitcast_convert) override;
  absl::Status HandleCall(HloInstruction* call) override;
  absl::Status HandleConvert(HloInstruction* convert) override;
  absl::Status HandleCopy(HloInstruction* copy) override;
  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  absl::Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleOptimizationBarrier(
      HloInstruction* optimization_barrier) override;
  absl::Status HandleReshape(HloInstruction* reshape) override;
  absl::Status HandleSlice(HloInstruction* slice) override;
  absl::Status HandleTranspose(HloInstruction* transpose) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleWhile(HloInstruction* xla_while) override;
  // go/keep-sorted end

 private:
  absl::Status HandleSimpleOp(HloInstruction* op);

 protected:
  HloDimensionAnalysis* analysis_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_DIMENSION_ANALYSIS_H_
