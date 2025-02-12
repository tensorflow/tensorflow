/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/dot_dimension_sorter.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/permutation_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {
namespace gpu {

namespace {

// Sort contracting dimensions of a dot() instruction preserving lhs-rhs pairs.
absl::Status SortDotDimensions(HloDotInstruction* dot) {
  const DotDimensionNumbers& dims = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dims(dims);
  new_dims.clear_lhs_contracting_dimensions();
  new_dims.clear_rhs_contracting_dimensions();
  const bool sort_by_lhs =
      DistinctNumbersAreConsecutiveIfSorted(dims.lhs_contracting_dimensions());
  // Sort lhs and rhs by sort_key using the fact that
  // sort_key is guaranteed to have only distinct consecutive numbers.
  const absl::Span<const int64_t>& sort_key =
      sort_by_lhs ? dims.lhs_contracting_dimensions()
                  : dims.rhs_contracting_dimensions();
  std::vector<int64_t> permutation;
  for (const int64_t a : sort_key) {
    permutation.push_back(a - *absl::c_min_element(sort_key));
  }
  const std::vector<int64_t> sorted_lhs =
      Permute(dims.lhs_contracting_dimensions(), permutation);
  *new_dims.mutable_lhs_contracting_dimensions() = {sorted_lhs.begin(),
                                                    sorted_lhs.end()};
  const std::vector<int64_t> sorted_rhs =
      Permute(dims.rhs_contracting_dimensions(), permutation);
  *new_dims.mutable_rhs_contracting_dimensions() = {sorted_rhs.begin(),
                                                    sorted_rhs.end()};
  std::unique_ptr<HloInstruction> new_dot = HloInstruction::CreateDot(
      dot->shape(), dot->mutable_operand(0), dot->mutable_operand(1), new_dims,
      dot->precision_config(), {dot->sparsity().begin(), dot->sparsity().end()},
      absl::MakeSpan(dot->operands()).subspan(HloDotInstruction::kOperands));
  dot->SetupDerivedInstruction(new_dot.get());

  VLOG(3) << "Sorted dot() dimensions:\n"
          << "\t before: " << dot->ToString() << "\n"
          << "\t after: " << new_dot->ToString();
  return dot->parent()->ReplaceWithNewInstruction(dot, std::move(new_dot));
}

}  // namespace

absl::StatusOr<bool> DotDimensionSorter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> dots_to_process;
  for (const HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : computation->instructions()) {
      if (HloPredicateIsNotOp<HloOpcode::kDot>(instr)) {
        continue;
      }
      // TODO(b/265688934): should non-default layouts be expected here at all?
      if ((instr->operand(0)->shape().has_layout() &&
           !LayoutUtil::IsMonotonicWithDim0Major(
               instr->operand(0)->shape().layout())) ||
          (instr->operand(1)->shape().has_layout() &&
           !LayoutUtil::IsMonotonicWithDim0Major(
               instr->operand(1)->shape().layout()))) {
        continue;
      }
      const DotDimensionNumbers& dims = instr->dot_dimension_numbers();
      if (dims.lhs_contracting_dimensions_size() == 0) {
        continue;
      }
      const bool cons_lhs = DistinctNumbersAreConsecutiveIfSorted(
          dims.lhs_contracting_dimensions());
      const bool cons_rhs = DistinctNumbersAreConsecutiveIfSorted(
          dims.rhs_contracting_dimensions());
      const bool sorted_lhs =
          absl::c_is_sorted(dims.lhs_contracting_dimensions());
      const bool sorted_rhs =
          absl::c_is_sorted(dims.rhs_contracting_dimensions());
      // The side to be sorted has to be consecutive and not sorted yet;
      // the other side should not get worsened.
      // TODO(b/265688934): we may still want to change which one is sorted
      // if this reduces the amount of transposed data.
      if ((cons_lhs && !sorted_lhs && !cons_rhs) ||
          (cons_rhs && !sorted_rhs && !cons_lhs) ||
          (cons_lhs && !sorted_lhs && cons_rhs && !sorted_rhs)) {
        dots_to_process.push_back(instr);
      }
    }
  }
  if (dots_to_process.empty()) {
    return false;
  }
  for (HloInstruction* dot : dots_to_process) {
    TF_RETURN_IF_ERROR(SortDotDimensions(Cast<HloDotInstruction>(dot)));
  }
  return true;
}

}  // namespace gpu
}  // namespace xla
