/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/reshape_decomposer.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/status.h"

namespace xla {

namespace {

class ReshapeDecomposerVisitor : public DfsHloRewriteVisitor {
 public:
  Status HandleReshape(HloInstruction* reshape) override {
    HloInstruction* operand = reshape->mutable_operand(0);
    auto s = reshape->shape();
    auto s0 = operand->shape();
    if (ShapeUtil::ReshapeIsBitcast(s, s0)) {
      return OkStatus();
    } else if (auto output_aligned_input_shape =
                   ShapeUtil::AlignLayouts(s, s0)) {
      Shape new_input_shape = *output_aligned_input_shape;
      HloInstruction* copied_operand = MakeCopyHlo(operand, new_input_shape);
      VLOG(3) << "Decomposing reshape into reshape-bitcast and a physical "
                 "transpose on the operand: "
              << copied_operand->ToString();

      TF_ASSIGN_OR_RETURN(auto r, MakeReshapeHlo(s, copied_operand));
      TF_RETURN_IF_ERROR(ReplaceInstruction(reshape, r));
      DCHECK(ShapeUtil::ReshapeIsBitcast(r->shape(), r->operand(0)->shape()));
    } else if (auto input_aligned_output_shape =
                   ShapeUtil::AlignLayouts(s0, s)) {
      Shape new_output_shape = *input_aligned_output_shape;
      TF_ASSIGN_OR_RETURN(auto r, MakeReshapeHlo(new_output_shape, operand));
      DCHECK(ShapeUtil::ReshapeIsBitcast(r->shape(), r->operand(0)->shape()));
      HloInstruction* copied_result = MakeCopyHlo(r, s);
      VLOG(3) << "Decomposing reshape into reshape-bitcast and a physical "
                 "transposition on the operand: "
              << copied_result->ToString();
      TF_RETURN_IF_ERROR(ReplaceInstruction(reshape, copied_result));
    } else {
      VLOG(3) << "Both input and output of reshape are not alignable, create "
                 "two physical transposes";
      auto s0_normalized = ShapeUtil::MakeShapeWithDescendingLayout(
          s0.element_type(), s0.dimensions());
      auto c1 = MakeCopyHlo(reshape->mutable_operand(0), s0_normalized);
      auto s_normalized = ShapeUtil::MakeShapeWithDescendingLayout(
          s.element_type(), s.dimensions());
      TF_ASSIGN_OR_RETURN(auto r, MakeReshapeHlo(s_normalized, c1));
      DCHECK(ShapeUtil::ReshapeIsBitcast(r->shape(), r->operand(0)->shape()));
      auto c2 = MakeCopyHlo(r, s);
      TF_RETURN_IF_ERROR(ReplaceInstruction(reshape, c2));
    }
    return OkStatus();
  }
};

}  // namespace

StatusOr<bool> ReshapeDecomposer::Run(HloModule* module) {
  return ReshapeDecomposerVisitor{}.RunOnModule(module);
}

}  // namespace xla
