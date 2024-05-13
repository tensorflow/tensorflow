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

#include "xla/service/gpu/gemm_degenerate_dim_remover.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

// Construct a new layout by adding removing the minor-most dimension to the
// input layout. For example, {3, 2, 1, 0} is extended to {2, 1, 0}.
// We expect that the input layout is normalized by LayoutNormalizer, so that
// the input layout has a descending ordering.
absl::StatusOr<Layout> GetLayoutWithNewMinorMostDimension(
    const Layout& layout) {
  if (!LayoutUtil::IsMonotonicWithDim0Major(layout)) {
    return absl::InvalidArgumentError("Layout is not normalized.");
  }
  return LayoutUtil::MakeDescendingLayout(layout.minor_to_major_size() - 1);
}

class GemmDegenerateDimRemoverVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleDot(HloInstruction* instr) override {
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);

    HloInstruction* new_lhs = nullptr;
    HloInstruction* new_rhs = nullptr;

    // The degenerate dimension is the last dimension of the LHS or RHS.
    if (lhs->shape().dimensions().back() == 1) {
      if (lhs->opcode() != HloOpcode::kBitcast) {
        return absl::InternalError("Degenerate operand is not a bitcast.");
      }
      new_lhs = lhs->mutable_operand(0);
      new_rhs = rhs;
    } else if (rhs->shape().dimensions().back() == 1) {
      if (rhs->opcode() != HloOpcode::kBitcast) {
        return absl::InternalError("Degenerate operand is not a bitcast.");
      }
      new_lhs = lhs;
      new_rhs = rhs->mutable_operand(0);
    } else {
      return absl::OkStatus();
    }

    changed_ = true;

    std::vector<int64_t> new_out_dimensions;
    new_out_dimensions.reserve(dot->shape().dimensions().size() - 1);
    for (int64_t dim_size : dot->shape().dimensions()) {
      if (dim_size == 1) {
        continue;
      }
      new_out_dimensions.push_back(dim_size);
    }

    // GemvRewriter should only add one degenerate dimension.
    if (new_out_dimensions.size() != dot->shape().dimensions().size() - 1) {
      return absl::InternalError(
          "More than one degenerate dimension in the output shape.");
    }

    Shape new_out_shape(
        dot->shape().element_type(), new_out_dimensions,
        absl::InlinedVector<bool, 4>(new_out_dimensions.size(), false),
        /*tuple_shapes=*/{});
    TF_ASSIGN_OR_RETURN(
        *new_out_shape.mutable_layout(),
        GetLayoutWithNewMinorMostDimension(dot->shape().layout()));

    HloComputation* computation = dot->parent();
    HloInstruction* new_dot =
        computation->AddInstruction(HloInstruction::CreateDot(
            new_out_shape, new_lhs, new_rhs, dot->dot_dimension_numbers(),
            dot->precision_config()));

    if (dot->user_count() != 1) {
      return absl::InternalError("Dot should have exactly one user.");
    }
    HloInstruction* bitcast = dot->users()[0];
    if (bitcast->opcode() != HloOpcode::kBitcast) {
      return absl::InternalError("Dot user should be a bitcast.");
    }
    return computation->ReplaceInstruction(bitcast, new_dot);
  }

  bool changed() const { return changed_; }

 private:
  bool changed_ = false;
};

}  // namespace

absl::StatusOr<bool> GemmDegenerateDimRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  GemmDegenerateDimRemoverVisitor visitor;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  }
  return visitor.changed();
}

}  // namespace gpu
}  // namespace xla
