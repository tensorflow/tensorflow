/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/dot_dimension_normalizer.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/analysis/shape_tracker.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

absl::Status NormalizeCategory(DotOperandDims* dims, ShapeTracker* tracker,
                               DotOperandDims::Category category) {
  std::optional<std::vector<int64_t>> permutation =
      dims->PermuteToConsecutive(category);
  if (permutation.has_value()) {
    RETURN_IF_ERROR(tracker->AppendTranspose(*permutation));
  }
  RETURN_IF_ERROR(dims->CollapseCategory(category, false));
  RETURN_IF_ERROR(tracker->AppendReshape(dims->shape().dimensions()));
  return absl::OkStatus();
}

absl::StatusOr<HloInstruction*> NormalizeOperand(
    DotOperandDims* dims, HloInstruction* operand,
    bool normalize_noncontracting) {
  // Walk up the chain of shape transformations (transpose, reshape, bitcast).
  HloInstruction* current = operand;
  std::vector<HloInstruction*> chain;
  while (current->opcode() == HloOpcode::kTranspose ||
         current->opcode() == HloOpcode::kReshape ||
         current->opcode() == HloOpcode::kBitcast) {
    chain.push_back(current);
    current = current->mutable_operand(0);
  }
  HloInstruction* head = current;

  // Build the tracker from head to operand.
  ShapeTracker tracker(head->shape());
  for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
    RETURN_IF_ERROR(tracker.AppendInstruction(*it));
  }

  // Normalize contracting dimensions.
  RETURN_IF_ERROR(
      NormalizeCategory(dims, &tracker, DotOperandDims::kContracting));

  // Normalize non-contracting dimensions if requested.
  if (normalize_noncontracting &&
      dims->Rank(DotOperandDims::kNonContracting) > 1) {
    RETURN_IF_ERROR(
        NormalizeCategory(dims, &tracker, DotOperandDims::kNonContracting));
  }

  return tracker.ToInstructionChain(head);
}

class NormalizerVisitor : public DfsHloRewriteVisitor {
 public:
  explicit NormalizerVisitor(bool normalize_noncontracting_dimensions)
      : normalize_noncontracting_dimensions_(
            normalize_noncontracting_dimensions) {}

  absl::Status HandleDot(HloInstruction* dot) override {
    ASSIGN_OR_RETURN(DotOperandDims lhs_dims,
                     DotOperandDims::FromDotOperand(dot, 0));
    ASSIGN_OR_RETURN(DotOperandDims rhs_dims,
                     DotOperandDims::FromDotOperand(dot, 1));

    bool contracting_normalization_needed =
        lhs_dims.Rank(DotOperandDims::kContracting) > 1;

    bool non_contracting_normalization_needed =
        normalize_noncontracting_dimensions_ &&
        (lhs_dims.Rank(DotOperandDims::kNonContracting) > 1 ||
         rhs_dims.Rank(DotOperandDims::kNonContracting) > 1);

    if (!contracting_normalization_needed &&
        !non_contracting_normalization_needed) {
      return absl::OkStatus();
    }

    ASSIGN_OR_RETURN(HloInstruction * normalized_lhs,
                     NormalizeOperand(&lhs_dims, dot->mutable_operand(0),
                                      normalize_noncontracting_dimensions_));
    ASSIGN_OR_RETURN(HloInstruction * normalized_rhs,
                     NormalizeOperand(&rhs_dims, dot->mutable_operand(1),
                                      normalize_noncontracting_dimensions_));
    ASSIGN_OR_RETURN(
        DotDimensionNumbers new_dnums,
        DotOperandDims::CreateDotDimensionNumbers(lhs_dims, rhs_dims));
    ASSIGN_OR_RETURN(Shape new_dot_shape,
                     DotOperandDims::ComputeOutputShape(
                         dot->shape().element_type(), lhs_dims, rhs_dims));
    HloInstruction* new_dot = dot->parent()->AddInstruction(
        HloInstruction::CreateDot(new_dot_shape, normalized_lhs, normalized_rhs,
                                  new_dnums, dot->precision_config()),
        &dot->metadata());

    if (new_dot->shape() != dot->shape()) {
      HloInstruction* reshape = dot->parent()->AddInstruction(
          HloInstruction::CreateReshape(dot->shape(), new_dot),
          &dot->metadata());
      dot->SetupDerivedInstruction(new_dot);
      dot->SetupDerivedInstruction(reshape);
      return ReplaceInstruction(dot, reshape);
    }
    dot->SetupDerivedInstruction(new_dot);
    return ReplaceInstruction(dot, new_dot);
  }

 private:
  bool normalize_noncontracting_dimensions_;
};

}  // namespace

absl::StatusOr<bool> DotDimensionNormalizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return NormalizerVisitor(normalize_noncontracting_dimensions_)
      .RunOnModule(module, execution_threads);
}

}  // namespace xla
