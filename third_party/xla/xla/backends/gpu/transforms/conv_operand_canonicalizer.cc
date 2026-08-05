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

#include "xla/backends/gpu/transforms/conv_operand_canonicalizer.h"

#include <cstdint>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {

// Checks if a constant literal is losslessly convertible to S8.
bool LiteralFitsInS8(const Literal& literal) {
  if (!literal.shape().IsArray()) {
    return false;
  }
  PrimitiveType type = literal.shape().element_type();
  if (type == S8) {
    return true;
  }

  absl::StatusOr<Literal> converted1 = literal.Convert(S8);
  if (!converted1.ok()) {
    return false;
  }

  absl::StatusOr<Literal> converted2 = converted1->Convert(type);
  if (!converted2.ok()) {
    return false;
  }

  return literal == *converted2;
}

// Canonicalizes an s32 convolution operand into `s32 convert(s8_node)`.
absl::StatusOr<HloInstruction*> CanonicalizeOperandToS8Convert(
    HloComputation* comp, HloInstruction* operand) {
  if (operand->shape().element_type() != S32) {
    return operand;
  }

  // 1. s32 constant -> s32 convert(s8 constant)
  if (operand->opcode() == HloOpcode::kConstant &&
      LiteralFitsInS8(operand->literal())) {
    absl::StatusOr<Literal> s8_literal = operand->literal().Convert(S8);
    if (!s8_literal.ok()) {
      return operand;
    }

    HloInstruction* s8_const = comp->AddInstruction(
        HloInstruction::CreateConstant(std::move(*s8_literal)));
    return comp->AddInstruction(
        HloInstruction::CreateConvert(operand->shape(), s8_const));
  }

  // 2. Redundant Convert: s32 convert(s32 convert(s8_src)) -> s32
  // convert(s8_src)
  if (operand->opcode() == HloOpcode::kConvert) {
    HloInstruction* src = operand->mutable_operand(0);
    if (src->opcode() == HloOpcode::kConvert &&
        src->operand(0)->shape().element_type() == S8) {
      return comp->AddInstruction(HloInstruction::CreateConvert(
          operand->shape(), src->mutable_operand(0)));
    }
  }

  // 3. Push s32 convert down through spatial/elementwise ops (Reshape,
  // Transpose, Broadcast, Pad, Slice, etc.): op(s32 convert(s8_src)) -> s32
  // convert(s8 op(s8_src))
  if (operand->operand_count() > 0) {
    HloInstruction* src = operand->mutable_operand(0);
    if (src->opcode() == HloOpcode::kConvert &&
        src->operand(0)->shape().element_type() == S8) {
      HloInstruction::InstructionVector s8_operands = operand->operands();
      s8_operands[0] = src->mutable_operand(0);

      // Handle pad value if operand is Pad
      if (operand->opcode() == HloOpcode::kPad && s8_operands.size() > 1) {
        HloInstruction* pad_val = s8_operands[1];
        if (pad_val->opcode() == HloOpcode::kConstant &&
            LiteralFitsInS8(pad_val->literal())) {
          auto s8_lit = pad_val->literal().Convert(S8);
          if (s8_lit.ok()) {
            s8_operands[1] = comp->AddInstruction(
                HloInstruction::CreateConstant(std::move(*s8_lit)));
          }
        } else if (pad_val->opcode() == HloOpcode::kConvert &&
                   pad_val->operand(0)->shape().element_type() == S8) {
          s8_operands[1] = pad_val->mutable_operand(0);
        } else {
          return operand;  // Cannot convert pad value to S8
        }
      }

      Shape s8_shape = ShapeUtil::ChangeElementType(operand->shape(), S8);
      HloInstruction* s8_op = comp->AddInstruction(
          operand->CloneWithNewOperands(s8_shape, s8_operands));
      return comp->AddInstruction(
          HloInstruction::CreateConvert(operand->shape(), s8_op));
    }
  }

  return operand;
}

}  // namespace

absl::StatusOr<bool> ConvOperandCanonicalizer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      if (instr->opcode() != HloOpcode::kConvolution) {
        continue;
      }

      for (int64_t i = 0; i < instr->operand_count(); ++i) {
        HloInstruction* operand = instr->mutable_operand(i);
        ASSIGN_OR_RETURN(HloInstruction * new_operand,
                         CanonicalizeOperandToS8Convert(comp, operand));
        if (new_operand != operand) {
          RETURN_IF_ERROR(instr->ReplaceOperandWith(i, new_operand));
          changed = true;
        }
      }
    }
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
