/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/hlo_element_type_converter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {

HloInstruction* ToElementType(HloInstruction* hlo, PrimitiveType type) {
  if (hlo->shape().element_type() != type) {
    Shape shape = ShapeUtil::ChangeElementType(hlo->shape(), type);
    hlo = hlo->parent()->AddInstruction(
        HloInstruction::CreateConvert(shape, hlo));
  }
  CHECK_EQ(hlo->shape().element_type(), type);
  return hlo;
}

bool HasOperandType(HloInstruction* hlo, PrimitiveType type) {
  for (HloInstruction* operand : hlo->operands()) {
    if (operand->shape().element_type() == type) {
      return true;
    }
  }
  return false;
}

// Finds out the Tuple Shape of the new instruction after converting the element
// type of the operands of the original instruction from `from_type` to
// `to_type`.
//
// This routine assumes the resulting `shape` of the original instruction is a
// non-nested tuple. This assumption is currently safe as only kTuple, kInfeed,
// kOutfeed, kCall, kCustomCall and kBatchNorm* HLO instructions can produce
// results with tuple shapes, and this routine is only called to convert the
// result shapes of kBatchNorm* HLO instructions, which are non-nested tuples.
Shape GetConvertedTupleShape(const Shape& shape, PrimitiveType from_type,
                             PrimitiveType to_type) {
  std::vector<Shape> new_tuple_subshapes;
  const int64_t n = ShapeUtil::TupleElementCount(shape);
  new_tuple_subshapes.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    Shape subshape = ShapeUtil::GetTupleElementShape(shape, i);
    CHECK(!subshape.IsTuple());
    if (subshape.element_type() == from_type) {
      subshape = ShapeUtil::ChangeElementType(subshape, to_type);
    }
    new_tuple_subshapes.push_back(subshape);
  }
  return ShapeUtil::MakeTupleShape(new_tuple_subshapes);
}

// Converts the elements of the result of `hlo` to produce a new tuple with
// shape `to_shape`.
//
// This routine assumes `hlo` is an instruction that produces a non-nested Tuple
// as a result.
HloInstruction* ConvertTupleElements(HloInstruction* hlo,
                                     const Shape& to_shape) {
  const Shape& shape = hlo->shape();
  HloComputation* computation = hlo->parent();
  std::vector<HloInstruction*> tuple_elements;
  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& ele_shape = ShapeUtil::GetTupleElementShape(shape, i);
    HloInstruction* element = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(ele_shape, hlo, i));
    const Shape& to_ele_shape = ShapeUtil::GetTupleElementShape(to_shape, i);
    CHECK(!ele_shape.IsTuple());
    if (ele_shape.element_type() != to_ele_shape.element_type()) {
      element = computation->AddInstruction(
          HloInstruction::CreateConvert(to_ele_shape, element));
    }
    tuple_elements.push_back(element);
  }
  return computation->AddInstruction(
      HloInstruction::CreateTuple(tuple_elements));
}

}  // namespace

HloElementTypeConverter::HloElementTypeConverter(
    PrimitiveType eliminate_type, PrimitiveType replace_with_type)
    : eliminate_type_(eliminate_type), replace_with_type_(replace_with_type) {}

// This routine converts the arithmetic operations in the given module that use
// eliminate_type_ to operations that use replace_with_type_.
absl::StatusOr<bool> HloElementTypeConverter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      3, "HloElementTypeConverter::Run(), before:\n" + module->ToString());

  if (eliminate_type_ == replace_with_type_) {
    return false;
  }

  HloCloneContext context(module);
  bool changed = false;
  for (auto* computation : module->computations(execution_threads)) {
    for (auto* hlo : computation->MakeInstructionPostOrder()) {
      const auto opcode = hlo->opcode();
      // These are ops where it does not make sense to convert them.
      if (opcode == HloOpcode::kParameter || opcode == HloOpcode::kConstant ||
          opcode == HloOpcode::kTuple || opcode == HloOpcode::kConvert ||
          opcode == HloOpcode::kBitcastConvert ||
          opcode == HloOpcode::kGetTupleElement ||
          opcode == HloOpcode::kInfeed || opcode == HloOpcode::kOutfeed) {
        continue;
      }

      // We cannot change a CustomCall since we have no way of adjusting the
      // called binary to expect the updated type.
      if (opcode == HloOpcode::kCustomCall) {
        continue;
      }

      // These are ops with embedded computations where it suffices to convert
      // the embedded computations instead of converting the ops themselves.
      if (opcode == HloOpcode::kWhile || opcode == HloOpcode::kCall ||
          opcode == HloOpcode::kAllReduce ||
          opcode == HloOpcode::kReduceScatter ||
          opcode == HloOpcode::kAllReduceStart ||
          opcode == HloOpcode::kFusion || opcode == HloOpcode::kMap ||
          opcode == HloOpcode::kReduce || opcode == HloOpcode::kReduceWindow ||
          opcode == HloOpcode::kScatter ||
          opcode == HloOpcode::kSelectAndScatter ||
          opcode == HloOpcode::kSort || opcode == HloOpcode::kConditional) {
        continue;
      }
      TF_RET_CHECK(hlo->called_computations().empty()) << hlo->ToString();

      bool nullary = hlo->operands().empty();
      bool wrong_element_type = hlo->shape().element_type() == eliminate_type_;
      bool should_eliminate_type = (nullary && wrong_element_type) ||
                                   HasOperandType(hlo, eliminate_type_);
      if (!should_eliminate_type) {
        // If this CHECK fires, then this was an instruction that does not take
        // the elimination type as an operand but it does return it. This pass
        // does not have a feature to change the output type in that case, so
        // instead of silently failing to eliminate the type, it fails loudly.
        TF_RET_CHECK(hlo->shape().element_type() != eliminate_type_);
        continue;
      }

      // Handle instructions that perform arithmetic operations and contain
      // operands with eliminate_type_.
      //
      // First, convert the operands with eliminate_type_ to operands with
      // replace_with_type_.
      std::vector<HloInstruction*> new_operands;
      const auto& operands = hlo->operands();
      new_operands.reserve(operands.size());
      for (HloInstruction* operand : operands) {
        if (operand->shape().element_type() == eliminate_type_) {
          operand = ToElementType(operand, replace_with_type_);
        }
        new_operands.push_back(operand);
      }

      // Then find out the result type of the new instruction with the same
      // opcode but using the converted operands, create the new instruction,
      // and convert the result of the new instruction back to match the result
      // type of the original instruction.
      HloInstruction* new_hlo;
      if (hlo->shape().element_type() == eliminate_type_) {
        Shape shape =
            ShapeUtil::ChangeElementType(hlo->shape(), replace_with_type_);

        new_hlo = computation->AddInstruction(
            hlo->CloneWithNewOperands(shape, new_operands, &context));
        TF_RETURN_IF_ERROR(new_hlo->CopyAllControlDepsFrom(hlo));

        new_hlo = ToElementType(new_hlo, eliminate_type_);
      } else if (hlo->shape().IsTuple()) {
        Shape old_shape = hlo->shape();
        Shape new_shape = GetConvertedTupleShape(hlo->shape(), eliminate_type_,
                                                 replace_with_type_);

        new_hlo = computation->AddInstruction(
            hlo->CloneWithNewOperands(new_shape, new_operands, &context));
        TF_RETURN_IF_ERROR(new_hlo->CopyAllControlDepsFrom(hlo));

        // Convert the elements of the result of `new_hlo` to produce a new
        // tuple with shape `old_shape`.
        new_hlo = ConvertTupleElements(new_hlo, old_shape);
      } else {
        new_hlo = computation->AddInstruction(
            hlo->CloneWithNewOperands(hlo->shape(), new_operands, &context));
        TF_RETURN_IF_ERROR(new_hlo->CopyAllControlDepsFrom(hlo));
      }

      TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(new_hlo));
      TF_RETURN_IF_ERROR(hlo->DropAllControlDeps());

      // NB!  We want to replace and remove side effecting instructions like Rng
      // as well so we can't rely HloComputation::ReplaceInstruction to reliably
      // remove the replaced instruction.
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(hlo));
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "HloElementTypeConverter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
