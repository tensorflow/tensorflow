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

#include "tensorflow/compiler/xla/service/hlo_constant_folding.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace {

template <PrimitiveType primitive_src_type, PrimitiveType primitive_dest_type>
static std::unique_ptr<Literal> ConvertIfTypesMatch(
    const Literal& src_literal) {
  CHECK_EQ(primitive_src_type, src_literal.shape().element_type());
  return LiteralUtil::Convert<
      typename primitive_util::PrimitiveTypeToNative<primitive_src_type>::type,
      typename primitive_util::PrimitiveTypeToNative<
          primitive_dest_type>::type>(src_literal);
}

template <PrimitiveType primitive_src_type>
static std::unique_ptr<Literal> ConvertIfDestTypeMatches(
    const Literal& src_literal, PrimitiveType primitive_dest_type) {
  switch (primitive_dest_type) {
#define CONVERT_IF_TYPES_MATCH(type) \
  case (type):                       \
    return ConvertIfTypesMatch<primitive_src_type, (type)>(src_literal);
    CONVERT_IF_TYPES_MATCH(PRED)
    CONVERT_IF_TYPES_MATCH(S8)
    CONVERT_IF_TYPES_MATCH(S32)
    CONVERT_IF_TYPES_MATCH(S64)
    CONVERT_IF_TYPES_MATCH(U8)
    CONVERT_IF_TYPES_MATCH(U32)
    CONVERT_IF_TYPES_MATCH(U64)
    CONVERT_IF_TYPES_MATCH(F32)
    CONVERT_IF_TYPES_MATCH(F64)
#undef CONVERT_IF_TYPES_MATCH
    // Other types are not yet supported.
    default:
      LOG(FATAL) << "Unimplemented: ConvertIfDestTypeMatches for type "
                 << PrimitiveType_Name(src_literal.shape().element_type());
  }
}

static std::unique_ptr<Literal> ConvertIfSrcTypeMatches(
    const Literal& src_literal, PrimitiveType primitive_dest_type) {
  switch (src_literal.shape().element_type()) {
#define CONVERT_IF_DEST_TYPE_MATCHES(type) \
  case (type):                             \
    return ConvertIfDestTypeMatches<(type)>(src_literal, primitive_dest_type);
    CONVERT_IF_DEST_TYPE_MATCHES(PRED)
    CONVERT_IF_DEST_TYPE_MATCHES(S8)
    CONVERT_IF_DEST_TYPE_MATCHES(S32)
    CONVERT_IF_DEST_TYPE_MATCHES(S64)
    CONVERT_IF_DEST_TYPE_MATCHES(U8)
    CONVERT_IF_DEST_TYPE_MATCHES(U32)
    CONVERT_IF_DEST_TYPE_MATCHES(U64)
    CONVERT_IF_DEST_TYPE_MATCHES(F32)
    CONVERT_IF_DEST_TYPE_MATCHES(F64)
#undef CONVERT_IF_DEST_TYPE_MATCHES
    // Other types are not yet supported.
    default:
      LOG(FATAL) << "Unimplemented: ConvertIfSrcTypeMatches for type "
                 << PrimitiveType_Name(src_literal.shape().element_type());
  }
}

}  // namespace

// ConstantFolderVisitor traverses the HLO computation and reduces certain
// constant graph sections, to literals.
class ConstantFolderVisitor : public DfsHloVisitorWithDefault {
 public:
  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleConcatenate(
      HloInstruction* concatenate,
      tensorflow::gtl::ArraySlice<HloInstruction*> operands) override;

  Status HandleConvert(HloInstruction* convert,
                       HloInstruction* operand) override;

  Status HandleReshape(HloInstruction* reshape) override;

  Status HandleSlice(HloInstruction* slice, HloInstruction* operand) override;

  Status HandleTranspose(HloInstruction* transpose) override;

  // Returns whether a constant folding operation has occurred.
  const bool changed() const { return changed_; }

  // Runs the visitor on a computation and returns whether any changes were
  // performed.
  static StatusOr<bool> Run(HloComputation* computation);

 private:
  ConstantFolderVisitor() = default;

  // Replaces the existing HLO instruction old_instruction, with a literal,
  // and marks the optimizer status as changed.
  // Returns the Status representing the result of the replace operation.
  Status ReplaceWithConstant(HloInstruction* old_instruction,
                             std::unique_ptr<Literal> literal) {
    TF_RETURN_IF_ERROR(old_instruction->parent()->ReplaceWithNewInstruction(
        old_instruction, HloInstruction::CreateConstant(std::move(literal))));
    changed_ = true;
    return Status::OK();
  }

  // Whether any constant folding operations have occurred.
  bool changed_ = false;
};

StatusOr<bool> ConstantFolderVisitor::Run(HloComputation* computation) {
  ConstantFolderVisitor visitor;
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

StatusOr<bool> HloConstantFolding::Run(HloModule* module) {
  XLA_VLOG_LINES(2,
                 "HloConstantFolding::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto& comp : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool result, ConstantFolderVisitor::Run(comp.get()));
    changed = changed || result;
  }
  XLA_VLOG_LINES(2, "HloConstantFolding::Run(), after:\n" + module->ToString());
  return changed;
}

Status ConstantFolderVisitor::HandleReshape(HloInstruction* reshape) {
  if (reshape->operand(0)->opcode() == HloOpcode::kConstant) {
    TF_ASSIGN_OR_RETURN(
        auto reshaped_literal,
        LiteralUtil::Reshape(reshape->operand(0)->literal(),
                             AsInt64Slice(reshape->shape().dimensions())));
    return ReplaceWithConstant(reshape, std::move(reshaped_literal));
  }
  return Status::OK();
}

Status ConstantFolderVisitor::HandleTranspose(HloInstruction* transpose) {
  if (transpose->operand(0)->opcode() == HloOpcode::kConstant) {
    auto transposed_literal = LiteralUtil::Transpose(
        transpose->operand(0)->literal(), transpose->dimensions());
    return ReplaceWithConstant(transpose, std::move(transposed_literal));
  }
  return Status::OK();
}

Status ConstantFolderVisitor::HandleConcatenate(
    HloInstruction* concatenate,
    tensorflow::gtl::ArraySlice<HloInstruction*> operands) {
  if (operands[0]->opcode() == HloOpcode::kConstant) {
    // If all the operands of a concatenate are constant, fold them into a
    // single constant tensor.
    // The result concatenate dimension is going to be the sum of all the
    // concatenate dimensions of the arrays taking part of the operation.
    int64 concat_dim = concatenate->dimensions()[0];
    const Shape& reference_shape = operands[0]->shape();
    CHECK(!ShapeUtil::IsTuple(reference_shape));
    int64 rank = ShapeUtil::Rank(reference_shape);
    std::vector<int64> concat_dimensions(reference_shape.dimensions().begin(),
                                         reference_shape.dimensions().end());
    if (concat_dim < 0) {
      concat_dim += rank;
    }
    for (int64 i = 1; i < operands.size(); ++i) {
      const Shape& operand_shape = operands[i]->shape();
      CHECK(!ShapeUtil::IsTuple(operand_shape));
      if (operands[i]->opcode() != HloOpcode::kConstant) {
        return Status::OK();
      }
      // Accumulate the concat dimension from all tensors taking part to the
      // operation.
      concat_dimensions[concat_dim] +=
          ShapeUtil::GetDimension(operand_shape, concat_dim);
    }

    auto literal = LiteralUtil::CreateFromDimensions(
        reference_shape.element_type(), concat_dimensions);
    std::vector<int64> source_indices(rank, 0);
    std::vector<int64> dest_indices(concat_dimensions.size(), 0);
    for (auto operand : operands) {
      const Shape& operand_shape = operand->shape();
      TF_RETURN_IF_ERROR(LiteralUtil::Copy(
          operand->literal(), source_indices, literal.get(), dest_indices,
          AsInt64Slice(operand_shape.dimensions())));
      dest_indices[concat_dim] +=
          ShapeUtil::GetDimension(operand_shape, concat_dim);
    }
    return ReplaceWithConstant(concatenate, std::move(literal));
  }
  return Status::OK();
}

Status ConstantFolderVisitor::HandleSlice(HloInstruction* slice,
                                          HloInstruction* operand) {
  if (operand->opcode() == HloOpcode::kConstant) {
    const Shape& shape = slice->shape();
    auto literal = LiteralUtil::CreateFromDimensions(
        shape.element_type(), AsInt64Slice(shape.dimensions()));
    std::vector<int64> dest_indices(slice->slice_starts().size(), 0);
    TF_RETURN_IF_ERROR(LiteralUtil::Copy(
        operand->literal(), slice->slice_starts(), literal.get(), dest_indices,
        AsInt64Slice(shape.dimensions())));
    TF_RETURN_IF_ERROR(ReplaceWithConstant(slice, std::move(literal)));
  }
  return Status::OK();
}

Status ConstantFolderVisitor::HandleConvert(HloInstruction* convert,
                                            HloInstruction* operand) {
  if (operand->opcode() == HloOpcode::kConstant) {
    const Literal& src_literal = operand->literal();
    std::unique_ptr<Literal> new_constant =
        ConvertIfSrcTypeMatches(src_literal, convert->shape().element_type());
    return ReplaceWithConstant(convert, std::move(new_constant));
  }
  return Status::OK();
}

}  // namespace xla
