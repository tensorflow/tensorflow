/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/layout_normalization.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/padding.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/stream_executor/lib/statusor.h"

namespace xla {
namespace {

// Layout normalization visitor. Aims to achieve the global postcondition that
// every layout is strictly descending (the layout permutation is effectively
// applied to the shape itself).
//
// Local precondition for every call:
//    -> Input is a bitcast from a normalized layout.
//
// Local postcondition:
//    -> Input and output of a processed operation have descending layout*
//
// *: For current fusion limitations this is currently not applicable to
// unnested reductions only.
class LayoutNormalizationVisitor : public DfsHloRewriteVisitor {
 public:
  // Default action: ensure local postcondition that any input is always a
  // bitcast from canonical layout for any rewrites of the HLO users.
  //
  // Bitcast to descending layout and then bitcast back to make sure that shapes
  // match.
  Status DefaultAction(HloInstruction* hlo) override {
    if (!hlo->user_count()) {
      // The local postcondition does not have to apply to the case when there
      // are no users.
      return OkStatus();
    }
    auto users = hlo->users();
    auto shape = hlo->shape();
    if (shape.IsTuple() || shape.IsToken()) {
      // GTEs will be transformed individually, tokens should be skipped.
      return OkStatus();
    }

    auto normalized_shape = Normalize(shape);
    auto bc_to_normalized = MakeBitcastHlo(hlo, normalized_shape);
    auto bc_to_orig = MakeBitcastHlo(bc_to_normalized, shape);
    TF_RETURN_IF_ERROR(hlo->ReplaceUsesWith(users, bc_to_orig));
    MarkAsChanged();
    return OkStatus();
  }

  // Pushes down the bitcast across the unary.
  // That is, converts:
  //
  //    H_0{I} -> B{L} -> U{L}
  //
  // into
  //
  //    H_0{I} -> U{I} -> B{L}
  //
  // where {I} denotes default layout.
  Status HandleElementwiseUnary(HloInstruction* hlo) override {
    if (hlo->opcode() == HloOpcode::kCopy) {
      // TODO(cheshire): Copy should not be really treated as elementwise.
      return DefaultAction(hlo);
    }
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    auto operand_shape = operand->shape();

    // Precondition: elementwise unary leaves layout intact.
    TF_RET_CHECK(s.layout() == operand_shape.layout())
        << "Unexpected non-layout preserving elementwise unary: "
        << hlo->ToString();
    TF_ASSIGN_OR_RETURN(auto normalized_input, GetNormalizedInput(operand));

    PrimitiveType to_element_type = s.element_type();
    HloInstruction* new_unary;
    if (hlo->opcode() == HloOpcode::kConvert) {
      new_unary = MakeConvertToHlo(normalized_input, to_element_type);
    } else if (hlo->opcode() == HloOpcode::kReducePrecision) {
      new_unary = MakeReducePrecisionHlo(normalized_input, hlo->exponent_bits(),
                                         hlo->mantissa_bits());
    } else if (hlo->opcode() == HloOpcode::kBitcastConvert) {
      new_unary = MakeBitcastConvertToHlo(normalized_input, to_element_type);
    } else {
      TF_ASSIGN_OR_RETURN(new_unary,
                          MakeUnaryHlo(hlo->opcode(), normalized_input));
    }
    auto bc_to_orig = MakeBitcastHlo(new_unary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // Pushes down the bitcast across the binary. Converts:
  //
  //  A1{I} -> bitcast{L}
  //            \
  //            B{L}
  //            /
  //  A2{I} -> bitcast{L}
  //
  // Into:
  //
  //  A1{I}
  //        \
  //         B{I} - bitcast{L}
  //        /
  //  A2{I}
  Status HandleElementwiseBinary(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto a = hlo->mutable_operand(0);
    auto b = hlo->mutable_operand(1);
    TF_RET_CHECK(a->shape().layout() == s.layout());
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(a));
    TF_ASSIGN_OR_RETURN(auto b0, GetNormalizedInput(b));

    HloInstruction* new_binary;
    if (hlo->opcode() == HloOpcode::kCompare) {
      TF_ASSIGN_OR_RETURN(new_binary,
                          MakeCompareHlo(hlo->comparison_direction(), a0, b0));
    } else {
      TF_ASSIGN_OR_RETURN(new_binary, MakeBinaryHlo(hlo->opcode(), a0, b0));
    }
    auto bc_to_orig = MakeBitcastHlo(new_binary, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

  // The ReshapeDecomposer already gives us a precondition that a reshape is
  // bitcast. Converts:
  //
  // A{I} -> bitcast [S0]{L1} -> R [S]{L2}
  //
  // Into:
  //
  // A{I} -> R [S']{I} -> bitcast[S]{L2}
  //
  Status HandleReshape(HloInstruction* hlo) override {
    auto s = hlo->shape();
    auto operand = hlo->mutable_operand(0);
    TF_RET_CHECK(ShapeUtil::ReshapeIsBitcast(s, operand->shape()));
    TF_ASSIGN_OR_RETURN(auto a0, GetNormalizedInput(operand));
    auto normalized_reshape_s = Normalize(s);
    TF_ASSIGN_OR_RETURN(auto new_reshape,
                        MakeReshapeHlo(normalized_reshape_s, a0));
    auto bc_to_orig = MakeBitcastHlo(new_reshape, s);
    TF_RETURN_IF_ERROR(ReplaceInstruction(hlo, bc_to_orig));
    return OkStatus();
  }

 private:
  // Due to Local Precondition we have, the input to all processed ops should
  // be HLO in descending layout piped through bitcast.
  StatusOr<HloInstruction*> GetNormalizedInput(HloInstruction* hlo) {
    TF_RET_CHECK(hlo->opcode() == HloOpcode::kBitcast)
        << "Unexpected HLO input: " << hlo->ToString();
    auto input = hlo->mutable_operand(0);
    auto input_shape = input->shape();
    TF_RET_CHECK(input_shape.layout() ==
                 LayoutUtil::GetDefaultLayoutForShape(input_shape));
    return input;
  }

  // Forces the layout to be descending and removes degenerate dimensions
  // without altering physical layout.
  Shape Normalize(const Shape& s) {
    return ShapeUtil::DropDegenerateDimensions(
        ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(s));
  }
};

}  // end namespace

StatusOr<bool> LayoutNormalization::Run(HloModule* module) {
  return LayoutNormalizationVisitor{}.RunOnModule(module);
}

}  // end namespace xla
