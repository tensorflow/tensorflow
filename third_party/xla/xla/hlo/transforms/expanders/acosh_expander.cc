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

// Expand acosh to MHLO dialect as follows:
//   acosh(x) = log(x + sqrt(x^2 - 1))      if x >= -1
//            = log(x + sqrt((x+1)*(x-1)))
//   acosh(x) = nan                         if x < -1
//
// If x^2 will overflow, we approximate sqrt(x^2 - 1) == x and compute as
// log(2*x) = log(2) + log(x).  (Note this works because negative x never
// overflows; x < -1 simply yields nan.
absl::StatusOr<HloInstruction*> AcoshExpander::ExpandInstruction(
    HloInstruction* instruction) {
  HloInstruction* x = instruction->mutable_operand(0);

  HloInstruction* one = MakeScalarLike(x, 1.0f);
  HloInstruction* two = MakeScalarLike(x, 2.0f);
  HloInstruction* max_finite_value = MakeScalarLikeFromLiteral(
      x, LiteralUtil::MaxFiniteValue(x->shape().element_type()));

  HloInstruction* div =
      MakeBinaryHlo(HloOpcode::kDivide, max_finite_value, two).value();
  HloInstruction* ge = MakeCompareHlo(ComparisonDirection::kGe, x, div).value();
  HloInstruction* log_two = MakeUnaryHlo(HloOpcode::kLog, two).value();
  HloInstruction* log_x = MakeUnaryHlo(HloOpcode::kLog, x).value();
  HloInstruction* log_sum =
      MakeBinaryHlo(HloOpcode::kAdd, log_two, log_x).value();
  HloInstruction* subtract =
      MakeBinaryHlo(HloOpcode::kSubtract, x, one).value();
  HloInstruction* sqrt = MakeUnaryHlo(HloOpcode::kSqrt, subtract).value();
  HloInstruction* x_plus_sqrt = MakeBinaryHlo(HloOpcode::kAdd, x, sqrt).value();
  HloInstruction* sqrt_2 = MakeUnaryHlo(HloOpcode::kSqrt, x_plus_sqrt).value();
  HloInstruction* sqrt_sum =
      MakeBinaryHlo(HloOpcode::kAdd, sqrt_2, sqrt).value();
  HloInstruction* sqrt_prod =
      MakeBinaryHlo(HloOpcode::kMultiply, sqrt, sqrt_sum).value();
  HloInstruction* log_1p = MakeUnaryHlo(HloOpcode::kLog1p, sqrt_prod).value();
  HloInstruction* select = MakeSelectHlo(ge, log_sum, log_1p).value();
  return select;
}

}  // namespace xla
