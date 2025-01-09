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

#include "xla/service/gpu/transforms/simplify_int4_dots.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/fusions/triton/triton_support.h"
#include "xla/service/pattern_matcher.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

namespace m = match;

absl::StatusOr<int> TranslateDimensionIdx(int dim_idx,
                                          const HloInstruction* bitcast) {
  int rank = bitcast->shape().rank();
  auto before = bitcast->operand(0)->shape().layout();
  auto after = bitcast->shape().layout();

  auto physical_location = after.minor_to_major(dim_idx);
  for (int idx = 0; idx < rank; ++idx) {
    if (before.minor_to_major(idx) == physical_location) {
      return idx;
    }
  }
  return absl::NotFoundError("failed to find physical location");
}

class SimplifyInt4DotsVisitor : public DfsHloRewriteVisitor {
 public:
  bool Run(HloComputation* computation, SimplifyInt4Dots* simplifier) {
    TF_CHECK_OK(computation->Accept(this));
    return changed();
  }

  // We handle the case where a bitcast is applied to a convert instruction that
  // converts from S4 to F32. We want to drop the bitcast and update the dot
  // dimension numbers to reflect the new layout. That happens when the layout
  // of the convert operand is not default. Example HLO:
  //
  // HloModule NonstandardLayoutInt4
  // ENTRY main {
  //   p0 = s4[64,128]{0,1} parameter(0)
  //   p1 = bf16[256,64]{1,0} parameter(1)
  //   ROOT %dot = bf16[128,256]{1,0} dot(s4[64,128]{0,1} p0, bf16[256,64]{1,0}
  //   p1),
  //     lhs_contracting_dims={0},
  //     rhs_contracting_dims={1}
  // }
  absl::Status HandleDot(HloInstruction* instr) override {
    HloDotInstruction* dot = Cast<HloDotInstruction>(instr);

    auto translate = [&](tsl::protobuf::RepeatedField<int64_t>* dims,
                         const HloInstruction* bitcast) -> absl::Status {
      for (auto& dim : *dims) {
        TF_ASSIGN_OR_RETURN(dim, TranslateDimensionIdx(dim, bitcast));
      }
      return absl::OkStatus();
    };

    const HloInstruction* arg;
    const HloInstruction* lhs = dot->operand(0);
    auto* dims = dot->mutable_dot_dimension_numbers();
    if (Match(lhs, m::Bitcast(m::Convert(&arg)))) {
      if (arg->operand(0)->shape().element_type() == S4) {
        TF_RETURN_IF_ERROR(
            translate(dims->mutable_lhs_contracting_dimensions(), lhs));
        TF_RETURN_IF_ERROR(
            translate(dims->mutable_lhs_batch_dimensions(), lhs));
        HloInstruction* convert = const_cast<HloInstruction*>(arg);
        TF_RETURN_IF_ERROR(dot->ReplaceOperandWithDifferentShape(0, convert));
        MarkAsChanged();
      }
    }

    const HloInstruction* rhs = dot->operand(1);
    if (Match(rhs, m::Bitcast(m::Convert(&arg)))) {
      if (arg->operand(0)->shape().element_type() == S4) {
        TF_RETURN_IF_ERROR(
            translate(dims->mutable_rhs_contracting_dimensions(), rhs));
        TF_RETURN_IF_ERROR(
            translate(dims->mutable_rhs_batch_dimensions(), rhs));
        HloInstruction* convert = const_cast<HloInstruction*>(arg);
        TF_RETURN_IF_ERROR(dot->ReplaceOperandWithDifferentShape(1, convert));
        MarkAsChanged();
      }
    }
    return absl::OkStatus();
  }
};

absl::StatusOr<bool> SimplifyInt4Dots::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  SimplifyInt4DotsVisitor visitor;
  for (HloComputation* comp :
       module->MakeComputationPostOrder(execution_threads)) {
    if (!IsTritonFusedComputation(*comp)) continue;
    changed |= visitor.Run(comp, this);
  }
  return changed;
}

}  // namespace xla::gpu
