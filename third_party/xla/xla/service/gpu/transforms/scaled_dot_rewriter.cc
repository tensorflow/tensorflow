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

#include "xla/service/gpu/transforms/scaled_dot_rewriter.h"

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

// Returns the type to use for the scaled dot operation operands.
// If both types are smaller than 16 bits, use BF16.
// If both types are the same, use the same type.
// Otherwise, use the bigger type.
PrimitiveType GetTargetType(PrimitiveType type_one, PrimitiveType type_two) {
  constexpr int kMinBitWidth = 16;
  if (primitive_util::BitWidth(type_one) < kMinBitWidth &&
      primitive_util::BitWidth(type_two) < kMinBitWidth) {
    return PrimitiveType::BF16;
  }
  if (type_one == type_two) {
    return type_one;
  }
  if (primitive_util::BitWidth(type_one) ==
          primitive_util::BitWidth(type_two) &&
      type_one != type_two) {
    return PrimitiveType::F32;
  }
  return (primitive_util::BitWidth(type_one) >
          primitive_util::BitWidth(type_two))
             ? type_one
             : type_two;
}

HloInstruction* Convert(HloInstruction* instr, PrimitiveType target_type) {
  if (instr->shape().element_type() == target_type) {
    return instr;
  }
  HloComputation* computation = instr->parent();
  Shape shape = instr->shape();
  shape.set_element_type(target_type);
  return computation->AddInstruction(
      HloInstruction::CreateConvert(shape, instr));
}

// Returns a pair of instructions that upscale both operands to the same type.
std::pair<HloInstruction*, HloInstruction*> UpscaleBoth(
    HloInstruction* first, HloInstruction* second) {
  PrimitiveType target_type = GetTargetType(first->shape().element_type(),
                                            second->shape().element_type());
  first = Convert(first, target_type);
  second = Convert(second, target_type);
  return std::make_pair(first, second);
}

absl::Status CheckOperandAndScaleShapes(absl::string_view side,
                                        const HloInstruction* operand,
                                        const HloInstruction* scale) {
  if (operand->shape().dimensions().size() !=
      scale->shape().dimensions().size()) {
    return InvalidArgument(
        "%s: operand and scale must have the same rank: %d vs %d", side,
        operand->shape().dimensions().size(),
        scale->shape().dimensions().size());
  }

  for (int i = 0; i < operand->shape().dimensions().size(); ++i) {
    if (operand->shape().dimensions(i) % scale->shape().dimensions(i)) {
      return InvalidArgument(
          "%s: operand and scale dimensions must match or scale dimension must "
          "be divider of operand dimension: %d vs %d at index %d",
          side, operand->shape().dimensions(i), scale->shape().dimensions(i),
          i);
    }
  }
  return absl::OkStatus();
}

HloInstruction* BroadcastAndReshape(HloInstruction* scale,
                                    const Shape& operand_shape,
                                    HloComputation* computation) {
  Shape scale_shape = scale->shape();
  std::vector<int64_t> broadcast_dims;
  std::vector<int64_t> shape_dims;

  for (int shape_index = 0, i = 0; i < operand_shape.dimensions().size();
       ++shape_index, ++i) {
    broadcast_dims.push_back(shape_index);
    shape_dims.push_back(scale_shape.dimensions(i));
    if (operand_shape.dimensions(i) != scale_shape.dimensions(i)) {
      ++shape_index;
      shape_dims.push_back(operand_shape.dimensions(i) /
                           scale_shape.dimensions(i));
    }
  }
  Shape new_scales_shape(scale_shape.element_type(), shape_dims);
  HloInstruction* new_scales = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_scales_shape, scale, broadcast_dims));
  Shape reshaped_scales_shape(scale_shape.element_type(),
                              operand_shape.dimensions());
  return computation->AddInstruction(
      HloInstruction::CreateReshape(reshaped_scales_shape, new_scales));
}

// Dequantizes the dot operation operand at the given index.
// The scale operand is expected to be broadcastable to the operand shape.
absl::StatusOr<HloInstruction*> Dequantize(HloInstruction* dot,
                                           int operand_index, int scale_index,
                                           absl::string_view side) {
  HloComputation* computation = dot->parent();
  HloInstruction* operand = dot->mutable_operand(operand_index);
  HloInstruction* scale = dot->mutable_operand(scale_index);
  if (scale->shape().dimensions().empty()) {
    // If the scale is a scalar, we don't need to do anything.
    return operand;
  }
  std::tie(operand, scale) = UpscaleBoth(operand, scale);
  TF_RETURN_IF_ERROR(CheckOperandAndScaleShapes(side, operand, scale));
  HloInstruction* broadcasted_scale =
      BroadcastAndReshape(scale, operand->shape(), computation);
  HloInstruction* dequantized =
      computation->AddInstruction(HloInstruction::CreateBinary(
          operand->shape(), HloOpcode::kMultiply, operand, broadcasted_scale));
  return dequantized;
}

// Returns true if the operand type is supported by the scaled dot operation.
// The source of truth is triton doc for dot_scaled operation.
// https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html#triton.language.dot_scaled
bool DoesTritonSupportOperandType(const HloInstruction* operand) {
  PrimitiveType type = operand->shape().element_type();
  switch (type) {
    case PrimitiveType::F16:
    case PrimitiveType::BF16:
    case PrimitiveType::F8E4M3:
    case PrimitiveType::F8E4M3FN:
    case PrimitiveType::F8E5M2:
    case PrimitiveType::F4E2M1FN:
      return true;
    default:
      return false;
  }
}

// Returns true if the operand type is supported by the scaled dot operation.
// The source of truth is triton doc for dot_scaled operation.
// https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html#triton.language.dot_scaled
bool DoesTritonSupportScaleType(const HloInstruction* scale) {
  PrimitiveType type = scale->shape().element_type();
  switch (type) {
    case PrimitiveType::F8E8M0FNU:
      return true;
    default:
      return false;
  }
}

bool IsRewriteRequired(HloInstruction* instruction) {
  if (!instruction->GetModule()
           ->config()
           .debug_options()
           .xla_gpu_experimental_enable_scaled_dot()) {
    return true;  // Always rewrite if the flag is not set.
  }
  HloScaledDotInstruction* dot = Cast<HloScaledDotInstruction>(instruction);
  if (!DoesTritonSupportOperandType(dot->operand(0))) {
    return true;
  }
  if (!DoesTritonSupportScaleType(dot->operand(1))) {
    return true;
  }
  if (!DoesTritonSupportOperandType(dot->operand(2))) {
    return true;
  }
  if (!DoesTritonSupportScaleType(dot->operand(3))) {
    return true;
  }

  return false;
}

}  // namespace

absl::StatusOr<bool> ScaledDotRewriter::Run(
    HloModule* module, const absl::flat_hash_set<absl::string_view>&) {
  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() != HloOpcode::kScaledDot) {
        continue;
      }
      if (!IsRewriteRequired(instruction)) {
        continue;
      }
      changed = true;
      HloScaledDotInstruction* dot = Cast<HloScaledDotInstruction>(instruction);
      TF_ASSIGN_OR_RETURN(HloInstruction * lhs, Dequantize(dot, 0, 1, "LHS"));
      TF_ASSIGN_OR_RETURN(HloInstruction * rhs, Dequantize(dot, 2, 3, "RHS"));

      TF_RETURN_IF_ERROR(dot->ReplaceAllUsesWith(
          computation->AddInstruction(HloInstruction::CreateDot(
              dot->shape(), lhs, rhs, dot->dot_dimension_numbers(),
              dot->precision_config()))));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(dot));
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
