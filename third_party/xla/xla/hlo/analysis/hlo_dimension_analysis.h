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

enum WeightInfo : uint8_t {
  kWeight,
  kTuple,
  kUnknown,
};

inline std::string WeightInfoToString(WeightInfo weight_info) {
  switch (weight_info) {
    case WeightInfo::kWeight:
      return "weight";
    case WeightInfo::kTuple:
      return "tuple";
    case WeightInfo::kUnknown:
      return "unknown";
  }
}

using WeightInfoMap =
    absl::node_hash_map<const HloInstruction*, ShapeTree<WeightInfo>>;

// This analysis pass determines which HLO instructions produce/are weights.
// Parameters to the entry computation are considered weights, and this property
// is propagated through instructions that preserve it (slice, convert,
// etc).
class HloDimensionAnalysis {
 public:
  friend class HloWeightPropagation;
  static absl::StatusOr<std::unique_ptr<HloDimensionAnalysis>> Run(
      const HloModule& module,
      const absl::flat_hash_set<absl::string_view>& execution_threads = {});

  // Whether the instruction has been annotated with weight info.
  bool HasWeightInfo(const HloInstruction* instruction) const {
    return info_map_.contains(instruction);
  }

  // Whether any leaf in the instruction shape is a weight.
  bool IsInstructionWeight(const HloInstruction* instruction) const;

  // Returns map of HLO instructions to their weight info.
  // If an instruction is not found in the map, it means that we have not
  // determined it is a weight.
  const WeightInfoMap& GetWeightInfoMap() const { return info_map_; }

  // Returns the weight info for the given instruction.
  std::optional<ShapeTree<WeightInfo>> GetWeightInfo(
      const HloInstruction* instruction) const;

 protected:
  explicit HloDimensionAnalysis(
      const HloModule& module,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      : module_(module), execution_threads_(execution_threads) {}

  // Sets the instruction as a weight. This is used to annotate the entry
  // computation parameters and other instructions that are known to be
  // weights.
  absl::Status SetInstructionAsWeight(HloInstruction* instruction);

  // Sets the weight info for the given target instruction.
  absl::Status SetWeightInfo(const HloInstruction* target,
                             ShapeTree<WeightInfo> weight_annotation);

  // Annotates the entry computation parameters as weights.
  absl::Status AnnotateEntryComputationParameters(const HloModule& module);

  // Runs the weight analysis on the given computation.
  absl::Status RunOnComputation(const HloComputation& computation);

  // Runs the weight analysis on the given computation, with the given operands
  // as the computation parameters. Propagates the weight info from the
  // operands to the computation parameters.
  absl::Status RunOnComputation(
      const HloComputation& computation,
      absl::Span<const HloInstruction* const> operands);

  WeightInfoMap info_map_;
  const HloModule& module_;
  const absl::flat_hash_set<absl::string_view>& execution_threads_;
};

class HloWeightPropagation : public DfsHloVisitorWithDefault {
 public:
  explicit HloWeightPropagation(HloDimensionAnalysis* dimension_analysis)
      : analysis_(dimension_analysis) {}
  absl::Status Run(const HloComputation& computation);
  absl::Status DefaultAction(HloInstruction* instruction) override;
  absl::Status HandleTuple(HloInstruction* tuple) override;
  absl::Status HandleGetTupleElement(
      HloInstruction* get_tuple_element) override;
  absl::Status HandleCall(HloInstruction* call) override;
  absl::Status HandleWhile(HloInstruction* xla_while) override;
  absl::Status HandleSimpleOp(HloInstruction* op);
  absl::Status HandleDynamicSlice(HloInstruction* dynamic_slice) override;
  absl::Status HandleDynamicUpdateSlice(
      HloInstruction* dynamic_update_slice) override;
  absl::Status HandleSlice(HloInstruction* slice) override;
  absl::Status HandleConvert(HloInstruction* convert) override;
  absl::Status HandleReshape(HloInstruction* reshape) override;
  absl::Status HandleBitcast(HloInstruction* bitcast) override;
  absl::Status HandleTranspose(HloInstruction* transpose) override;
  absl::Status HandleCopy(HloInstruction* copy) override;
  absl::Status HandleBitcastConvert(HloInstruction* bitcast_convert) override;
  absl::Status HandleOptimizationBarrier(
      HloInstruction* optimization_barrier) override;

 protected:
  HloDimensionAnalysis* analysis_;
};

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_HLO_DIMENSION_ANALYSIS_H_
