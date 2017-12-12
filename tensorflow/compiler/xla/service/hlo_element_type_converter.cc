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

#include "tensorflow/compiler/xla/service/hlo_element_type_converter.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_evaluator.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/errors.h"

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

}  // namespace

HloElementTypeConverter::HloElementTypeConverter(
    PrimitiveType eliminate_type, PrimitiveType replace_with_type)
    : eliminate_type_(eliminate_type), replace_with_type_(replace_with_type) {}

StatusOr<bool> HloElementTypeConverter::Run(HloModule* module) {
  XLA_VLOG_LINES(
      3, "HloElementTypeConverter::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* computation : module->computations()) {
    for (auto* hlo : computation->MakeInstructionPostOrder()) {
      // These are ops where it does not make sense to convert them.
      if (hlo->opcode() == HloOpcode::kParameter ||
          hlo->opcode() == HloOpcode::kConstant ||
          hlo->opcode() == HloOpcode::kTuple ||
          hlo->opcode() == HloOpcode::kConvert ||
          hlo->opcode() == HloOpcode::kGetTupleElement ||
          hlo->opcode() == HloOpcode::kInfeed ||
          hlo->opcode() == HloOpcode::kOutfeed) {
        continue;
      }

      // We cannot change a CustomCall since we have no way of adjusting the
      // called binary to expect the updated type.
      if (hlo->opcode() == HloOpcode::kCustomCall) {
        continue;
      }

      // These are ops with embedded computations where it suffices to convert
      // the embedded computations instead of converting the ops themselves.
      if (hlo->opcode() == HloOpcode::kWhile ||
          hlo->opcode() == HloOpcode::kCall ||
          hlo->opcode() == HloOpcode::kFusion ||
          hlo->opcode() == HloOpcode::kMap ||
          hlo->opcode() == HloOpcode::kReduce ||
          hlo->opcode() == HloOpcode::kReduceWindow ||
          hlo->opcode() == HloOpcode::kSelectAndScatter ||
          hlo->opcode() == HloOpcode::kConditional) {
        continue;
      }
      TF_RET_CHECK(hlo->called_computations().empty()) << hlo->ToString();

      if (!HasOperandType(hlo, eliminate_type_)) {
        // If this CHECK fires, then this was an instruction that does not take
        // the elimination type as an operand but it does return it. This pass
        // does not have a feature to change the output type in that case, so
        // instead of silently failing to eliminate the type, it fails loudly.
        TF_RET_CHECK(hlo->shape().element_type() != eliminate_type_);
        continue;
      }

      std::vector<HloInstruction*> new_operands;
      for (HloInstruction* operand : hlo->operands()) {
        if (operand->shape().element_type() == eliminate_type_) {
          operand = ToElementType(operand, replace_with_type_);
        }
        new_operands.push_back(operand);
      }

      HloInstruction* new_hlo;
      if (hlo->shape().element_type() == eliminate_type_) {
        Shape shape =
            ShapeUtil::ChangeElementType(hlo->shape(), replace_with_type_);
        new_hlo = computation->AddInstruction(
            hlo->CloneWithNewOperands(shape, new_operands, hlo->GetModule()));
        new_hlo = ToElementType(new_hlo, eliminate_type_);
      } else {
        new_hlo = computation->AddInstruction(hlo->CloneWithNewOperands(
            hlo->shape(), new_operands, hlo->GetModule()));
      }
      TF_RETURN_IF_ERROR(computation->ReplaceInstruction(hlo, new_hlo));
      changed = true;
    }
  }
  XLA_VLOG_LINES(
      2, "HloElementTypeConverter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
