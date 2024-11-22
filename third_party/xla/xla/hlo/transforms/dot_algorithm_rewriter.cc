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

#include "xla/hlo/transforms/dot_algorithm_rewriter.h"

#include <cstdint>
#include <limits>
#include <tuple>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "tsl/platform/status.h"

namespace xla {

namespace {

HloInstruction* Truncate(HloInstruction* f32_param) {
  // Cast to int32 first, then zero out the high bits. Then cast back to f32.
  Shape u32_shape = f32_param->shape();
  u32_shape.set_element_type(PrimitiveType::U32);
  HloInstruction* u32_param = f32_param->AddInstruction(
      HloInstruction::CreateBitcastConvert(u32_shape, f32_param));
  HloInstruction* mask_constant =
      f32_param->parent()->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<uint32_t>(0xFFFF0000)));
  HloInstruction* u32_mask = u32_param->AddInstruction(
      HloInstruction::CreateBroadcast(u32_shape, mask_constant, {}));
  HloInstruction* masked_u32 =
      u32_param->AddInstruction(HloInstruction::CreateBinary(
          u32_shape, HloOpcode::kAnd, u32_param, u32_mask));
  return masked_u32->AddInstruction(
      HloInstruction::CreateBitcastConvert(f32_param->shape(), masked_u32));
}

HloInstruction* Sub(HloInstruction* instr, HloInstruction* high) {
  return instr->AddInstruction(HloInstruction::CreateBinary(
      instr->shape(), HloOpcode::kSubtract, instr, high));
}

HloInstruction* RoundToBF16(HloInstruction* instr) {
  Shape new_shape = instr->shape();
  new_shape.set_element_type(PrimitiveType::BF16);
  return instr->AddInstruction(HloInstruction::CreateConvert(new_shape, instr));
}

std::pair<HloInstruction*, HloInstruction*> Split2xToBF16(
    HloInstruction* f32_param) {
  HloInstruction* high_f32 = Truncate(f32_param);
  HloInstruction* low_f32 = Sub(f32_param, high_f32);
  return std::make_pair(RoundToBF16(high_f32), RoundToBF16(low_f32));
}

std::tuple<HloInstruction*, HloInstruction*, HloInstruction*> Split3xToBF16(
    HloInstruction* f32_param) {
  HloInstruction* high_f32_t = Truncate(f32_param);
  HloInstruction* mid_f32 = Sub(f32_param, high_f32_t);
  HloInstruction* mid_f32_t = Truncate(mid_f32);
  HloInstruction* low_f32 = Sub(mid_f32, mid_f32_t);
  return std::make_tuple(RoundToBF16(high_f32_t), RoundToBF16(mid_f32_t),
                         RoundToBF16(low_f32));
}

// If lhs is 1.0, we will have lhs_high = 1.0 and lhs_low = 0.0.
// If rhs is +infinity, we will have:
// +infinity * 1.0 = +infinity
// +infinity * 0.0 = NaN
// We would get the wrong result if we sum these partial products. Instead, we
// must override any accumulated result if the last partial product is
// non-finite. See b/115844437.
HloInstruction* ReplaceNaNWithZeros(HloInstruction* input) {
  HloComputation* computation = input->parent();
  Shape shape = input->shape();
  HloInstruction* infinity = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(
          std::numeric_limits<float>::infinity())));
  // Broadcast the infinity to the same shape as the result.
  infinity = computation->AddInstruction(
      HloInstruction::CreateBroadcast(shape, infinity, {}));
  // abs the result.
  HloInstruction* abs_result = computation->AddInstruction(
      HloInstruction::CreateUnary(shape, HloOpcode::kAbs, input));
  // Compare the abs result with the infinity.
  HloInstruction* cmp_result =
      computation->AddInstruction(HloInstruction::CreateCompare(
          shape, infinity, abs_result, ComparisonDirection::kGe));
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  zero = computation->AddInstruction(
      HloInstruction::CreateBroadcast(shape, zero, {}));
  // Select the high high dot if the result is less than the infinity.
  input = computation->AddInstruction(HloInstruction::CreateTernary(
      shape, HloOpcode::kSelect, cmp_result, input, zero));
  return input;
}

void RewriteF32ToBF16X3(HloInstruction* instr) {
  HloComputation* computation = instr->parent();
  HloDotInstruction* original_dot = Cast<HloDotInstruction>(instr);
  PrecisionConfig precision_config = original_dot->precision_config();
  precision_config.clear_algorithm();
  const Shape& shape = original_dot->shape();
  const DotDimensionNumbers& dnums = original_dot->dot_dimension_numbers();
  auto dot = [&](HloInstruction* lhs, HloInstruction* rhs) {
    return computation->AddInstruction(
        HloInstruction::CreateDot(shape, lhs, rhs, dnums, precision_config));
  };
  auto sum = [&](HloInstruction* lhs, HloInstruction* rhs) {
    return computation->AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));
  };

  auto [lhs_high_bf16, lhs_low_bf16] =
      Split2xToBF16(original_dot->mutable_operand(0));
  auto [rhs_high_bf16, rhs_low_bf16] =
      Split2xToBF16(original_dot->mutable_operand(1));

  HloInstruction* low_high_dot = dot(lhs_low_bf16, rhs_high_bf16);
  HloInstruction* high_low_dot = dot(lhs_high_bf16, rhs_low_bf16);
  HloInstruction* high_high_dot = dot(lhs_high_bf16, rhs_high_bf16);
  HloInstruction* low_sum = sum(low_high_dot, high_low_dot);
  low_sum = ReplaceNaNWithZeros(low_sum);
  HloInstruction* result = sum(low_sum, high_high_dot);
  TF_CHECK_OK(original_dot->ReplaceAllUsesWith(result));
  TF_CHECK_OK(original_dot->parent()->RemoveInstruction(original_dot));
}

void RewriteF32ToBF16X6(HloInstruction* instr) {
  HloComputation* computation = instr->parent();
  HloDotInstruction* original_dot = Cast<HloDotInstruction>(instr);
  PrecisionConfig precision_config = original_dot->precision_config();
  precision_config.clear_algorithm();
  const Shape& shape = original_dot->shape();
  const DotDimensionNumbers& dnums = original_dot->dot_dimension_numbers();
  auto dot = [&](HloInstruction* lhs, HloInstruction* rhs) {
    return computation->AddInstruction(
        HloInstruction::CreateDot(shape, lhs, rhs, dnums, precision_config));
  };
  auto sum = [&](HloInstruction* lhs, HloInstruction* rhs) {
    return computation->AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, lhs, rhs));
  };

  auto [lhs_high_bf16, lhs_mid_bf16, lhs_low_bf16] =
      Split3xToBF16(original_dot->mutable_operand(0));
  auto [rhs_high_bf16, rhs_mid_bf16, rhs_low_bf16] =
      Split3xToBF16(original_dot->mutable_operand(1));

  HloInstruction* middle_middle_dot = dot(lhs_mid_bf16, rhs_mid_bf16);
  HloInstruction* high_low_dot = dot(lhs_high_bf16, rhs_low_bf16);
  HloInstruction* low_high_dot = dot(lhs_low_bf16, rhs_high_bf16);
  HloInstruction* high_middle_dot = dot(lhs_high_bf16, rhs_mid_bf16);
  HloInstruction* middle_high_dot = dot(lhs_mid_bf16, rhs_high_bf16);
  HloInstruction* high_high_dot = dot(lhs_high_bf16, rhs_high_bf16);

  HloInstruction* result = nullptr;
  result = sum(middle_middle_dot, high_low_dot);
  result = sum(result, low_high_dot);
  result = sum(result, high_middle_dot);
  result = sum(result, middle_high_dot);
  result = ReplaceNaNWithZeros(result);
  result = sum(result, high_high_dot);

  TF_CHECK_OK(original_dot->ReplaceAllUsesWith(result));
  TF_CHECK_OK(original_dot->parent()->RemoveInstruction(original_dot));
}

}  // namespace

absl::StatusOr<bool> DotAlgorithmRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() != HloOpcode::kDot) {
        continue;
      }
      auto algorithm = instruction->precision_config().algorithm();
      switch (algorithm) {
        case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X3:
          RewriteF32ToBF16X3(instruction);
          changed = true;
          break;
        case PrecisionConfig::ALG_DOT_BF16_BF16_F32_X6:
          RewriteF32ToBF16X6(instruction);
          changed = true;
          break;
        default:
          break;
      }
    }
  }
  return changed;
}

}  // namespace xla
