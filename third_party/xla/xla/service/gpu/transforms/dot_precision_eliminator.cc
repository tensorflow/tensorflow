/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/gpu/transforms/dot_precision_eliminator.h"

#include <algorithm>
#include <cstddef>
#include <tuple>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

HloInstruction* AsType(HloInstruction* operand, PrimitiveType type) {
  if (operand->shape().element_type() == type) {
    return operand;
  }
  HloInstruction* convert =
      operand->parent()->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(operand->shape(), type), operand));
  return convert;
}

bool IsMorePrecise(PrimitiveType lhs, PrimitiveType rhs) {
  CHECK(primitive_util::IsFloatingPointType(lhs));
  CHECK(primitive_util::IsFloatingPointType(rhs));
  return std::make_tuple(primitive_util::ExponentWidth(lhs),
                         primitive_util::SignificandWidth(lhs)) >
         std::make_tuple(primitive_util::ExponentWidth(rhs),
                         primitive_util::SignificandWidth(rhs));
}

template <typename F>
bool EitherOperandType(HloInstruction* instruction, F&& pred) {
  return pred(instruction->operand(0)->shape().element_type()) ||
         pred(instruction->operand(1)->shape().element_type());
}

absl::StatusOr<bool> EliminateDotPrecisions(HloInstruction* instruction) {
  auto& precision_config = *instruction->mutable_precision_config();

  int highest_operand_precision = 0;
  if (precision_config.operand_precision_size() > 0) {
    highest_operand_precision =
        *std::max_element(precision_config.operand_precision().begin(),
                          precision_config.operand_precision().end());
  }
  if (highest_operand_precision == PrecisionConfig::DEFAULT) {
    return false;
  }

  // If any of the operands is complex type, return because we don't have
  // explicit algorithm for complex types yet.
  if (EitherOperandType(instruction, [](PrimitiveType type) {
        return primitive_util::IsComplexType(type);
      })) {
    return false;
  }

  // We have non-default precision. We reset it, so this function will
  // return true.
  precision_config.clear_operand_precision();
  precision_config.mutable_operand_precision()->Resize(
      instruction->operand_count(), PrecisionConfig::DEFAULT);

  // If both operands are not floating point, just remove operand precision.
  if (EitherOperandType(instruction, [](PrimitiveType type) {
        return !primitive_util::IsFloatingPointType(type);
      })) {
    return true;
  }

  // If operands are F32 and precision is HIGHEST, set algorithm to
  // dot_f32_f32_f32.
  if (EitherOperandType(instruction,
                        [](PrimitiveType type) { return type == F32; })) {
    if (highest_operand_precision == PrecisionConfig::HIGHEST) {
      precision_config.set_algorithm(PrecisionConfig::ALG_DOT_F32_F32_F32);
    }
    return true;
  }

  // If neither of operands is less precise than F32, just return.
  if (!EitherOperandType(instruction, [](PrimitiveType type) {
        return IsMorePrecise(F32, type);
      })) {
    return true;
  }

  // Upcast operands to F32 unless they are already F32.
  for (size_t i = 0; i < instruction->operand_count(); ++i) {
    auto* operand = instruction->mutable_operand(i);
    if (operand->shape().element_type() != F32) {
      TF_RETURN_IF_ERROR(
          instruction->ReplaceOperandWith(i, AsType(operand, F32)));
    }
  }

  // If dot output is less precise than operands, make it F32 and downcast.
  if (IsMorePrecise(F32, instruction->shape().element_type())) {
    auto* convert =
        instruction->parent()->AddInstruction(HloInstruction::CreateConvert(
            ShapeUtil::ChangeElementType(instruction->shape(),
                                         instruction->shape().element_type()),
            instruction));
    TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(convert));
    instruction->mutable_shape()->set_element_type(F32);
  }

  return true;
}

}  // namespace

absl::StatusOr<bool> DotPrecisionEliminator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kDot) {
        TF_ASSIGN_OR_RETURN(bool instruction_changed,
                            EliminateDotPrecisions(instruction));
        changed |= instruction_changed;
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
