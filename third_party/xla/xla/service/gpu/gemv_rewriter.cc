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

#include "xla/service/gpu/gemv_rewriter.h"

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/protobuf/repeated_field.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

class GemvRewriterVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleDot(HloInstruction* instr) override {
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);
    const DotDimensionNumbers& dim_numbers = dot->dot_dimension_numbers();
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);

    // This pass relies on dot decomposer which ensures that all non-batch
    // dimensions are merged to one.
    bool lhs_has_non_contracting_dim =
        lhs->shape().rank() ==
        dim_numbers.lhs_batch_dimensions_size() +
            dim_numbers.lhs_contracting_dimensions_size() + 1;
    bool rhs_has_non_contracting_dim =
        rhs->shape().rank() ==
        dim_numbers.rhs_batch_dimensions_size() +
            dim_numbers.rhs_contracting_dimensions_size() + 1;

    // Skip vector-vector multiplication.
    if (!lhs_has_non_contracting_dim && !rhs_has_non_contracting_dim) {
      return absl::OkStatus();
    }

    if (dot->shape().is_dynamic()) return absl::OkStatus();

    changed_ = true;

    HloComputation* computation = dot->parent();
    DotDimensionNumbers new_dim_numbers = dot->dot_dimension_numbers();
    HloInstruction* new_lhs = lhs;
    if (!lhs_has_non_contracting_dim) {
      std::vector<int64_t> new_dimensions;
      new_dimensions.reserve(lhs->shape().dimensions().size() + 1);
      for (int64_t dim_size : lhs->shape().dimensions()) {
        new_dimensions.push_back(dim_size);
      }
      new_dimensions.push_back(1);
      std::vector<const bool> dynamic_dimensions(new_dimensions.size(), false);
      Shape new_shape(lhs->shape().element_type(), new_dimensions,
                      dynamic_dimensions, /*tuple_shapes=*/{});
      new_lhs = computation->AddInstruction(
          HloInstruction::CreateBitcast(new_shape, lhs));
    }

    HloInstruction* new_rhs = rhs;
    if (!rhs_has_non_contracting_dim) {
      std::vector<int64_t> new_dimensions;
      new_dimensions.reserve(rhs->shape().dimensions().size() + 1);
      for (int64_t dim_size : rhs->shape().dimensions()) {
        new_dimensions.push_back(dim_size);
      }
      new_dimensions.push_back(1);
      std::vector<const bool> dynamic_dimensions(new_dimensions.size(), false);
      Shape new_shape(lhs->shape().element_type(), new_dimensions,
                      dynamic_dimensions, /*tuple_shapes=*/{});
      new_rhs = computation->AddInstruction(
          HloInstruction::CreateBitcast(new_shape, rhs));
    }

    std::vector<int64_t> new_dimensions;
    new_dimensions.reserve(dot->shape().dimensions().size() + 1);
    for (int64_t dim_size : dot->shape().dimensions()) {
      new_dimensions.push_back(dim_size);
    }
    if (!lhs_has_non_contracting_dim) {
      // Insert the trivial dimension before the non-contracting dimension from
      // rhs.
      int non_contracting_dim_size = new_dimensions.back();
      new_dimensions[new_dimensions.size() - 1] = 1;
      new_dimensions.push_back(non_contracting_dim_size);
    } else {
      new_dimensions.push_back(1);
    }

    std::vector<const bool> dynamic_dimensions(new_dimensions.size(), false);
    Shape new_shape(dot->shape().element_type(), new_dimensions,
                    dynamic_dimensions, /*tuple_shapes=*/{});
    HloInstruction* new_dot = computation->AddInstruction(
        HloInstruction::CreateDot(new_shape, new_lhs, new_rhs, new_dim_numbers,
                                  dot->precision_config()));
    HloInstruction* bitcast = computation->AddInstruction(
        HloInstruction::CreateBitcast(dot->shape(), new_dot));
    return computation->ReplaceInstruction(dot, bitcast);
  }

  bool changed() const { return changed_; }

 private:
  bool changed_ = false;
};

absl::StatusOr<bool> RunOnComputation(HloComputation* computation) {
  GemvRewriterVisitor gemv_rewriter;
  TF_RETURN_IF_ERROR(computation->Accept(&gemv_rewriter));
  return gemv_rewriter.changed();
}

}  // namespace

absl::StatusOr<bool> GemvRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
