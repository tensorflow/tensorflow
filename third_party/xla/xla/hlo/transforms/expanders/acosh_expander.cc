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

#include "xla/hlo/transforms/expanders/acosh_expander.h"

#include <limits>

#include "absl/status/statusor.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

bool AcoshExpander::InstructionMatchesPattern(HloInstruction* instruction) {
  return instruction->opcode() == HloOpcode::kAcosh;
}

//   nan                        if x < -1
//   log(x) + log(2)            if x >= sqrt_max_value
//   log(x + sqrt((x+1)*(x-1))) otherwise
absl::StatusOr<HloInstruction*> AcoshExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloInstruction* x = instruction->mutable_operand(0);
  const Shape shape = x->shape();
  PrimitiveType element_type = shape.element_type();

  HloInstruction* one = MakeScalarLike(x, 1.0f);
  HloInstruction* neg_one = MakeScalarLike(x, 1.0f);
  HloInstruction* two = MakeScalarLike(x, 1.0f);
  HloInstruction* nan =
      MakeScalarLike(x, std::numeric_limits<float>::quiet_NaN());

  // naive_result = Log(x + Sqrt((x + one) * (x - one)));
  HloInstruction* x_plus_one = MakeBinaryHlo(HloOpcode::kAdd, x, one).value();
  HloInstruction* x_min_one =
      MakeBinaryHlo(HloOpcode::kSubtract, x, one).value();
  HloInstruction* x_sq_min_one =
      MakeBinaryHlo(HloOpcode::kMultiply, x_plus_one, x_min_one).value();
  HloInstruction* sqrt = MakeUnaryHlo(HloOpcode::kSqrt, x_sq_min_one).value();
  HloInstruction* x_plus_sqrt = MakeBinaryHlo(HloOpcode::kAdd, x, sqrt).value();
  HloInstruction* naive_result =
      MakeUnaryHlo(HloOpcode::kLog, x_plus_sqrt).value();

  if (primitive_util::IsComplexType(element_type)) {
    return naive_result;
  }
  HloInstruction* log_x = MakeUnaryHlo(HloOpcode::kLog, x).value();
  HloInstruction* log_two = MakeUnaryHlo(HloOpcode::kLog, two).value();
  HloInstruction* overflow_result =
      MakeBinaryHlo(HloOpcode::kAdd, log_x, log_two).value();

  HloInstruction* max_finite_value =
      instruction->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::MaxFiniteValue(element_type)));
  HloInstruction* max_value_bcast = instruction->AddInstruction(
      HloInstruction::CreateBroadcast(shape, max_finite_value, {}));
  HloInstruction* sqrt_max_value =
      MakeUnaryHlo(HloOpcode::kSqrt, max_value_bcast).value();

  HloInstruction* ge =
      MakeCompareHlo(ComparisonDirection::kGe, x, sqrt_max_value).value();
  HloInstruction* select =
      MakeSelectHlo(ge, overflow_result, naive_result).value();

  HloInstruction* lt =
      MakeCompareHlo(ComparisonDirection::kLt, x, neg_one).value();
  HloInstruction* select2 = MakeSelectHlo(lt, nan, select).value();
  return select2;
}

}  // namespace xla
