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

#include "xla/hlo/transforms/expanders/elementary_function_expander.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/builder/lib/constants.h"
#include "xla/hlo/builder/xla_builder.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/call_inliner.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

// For f64:
//   Sinh(x) = sign(x) * (t - t^2 / (2 * (1 + t)))                  if |x| < 1
//           = sign(x) * (e^(|x| - c) +
//                        (e^(|x| - c) * s_lo - e^(-c - |x|) * s)) if |x| >= 1
// where t = expm1(|x|), c = 0x1.63p-1 (~0.693359375), s = e^c / 2, and
// s_lo = s - 1 = 0x1.bd0d1c3050e04p-13.
//
// For complex types (c64, c128), the shifted-exponential formula is used for
// all inputs.
XlaOp ExpandSinh(XlaOp x, PrimitiveType type) {
  ResultAccuracy highest_accuracy;
  highest_accuracy.set_mode(ResultAccuracy::HIGHEST);

  // Computing (e^x - e^-x) / 2 directly overflows for large |x| when e^|x| =
  // inf but (e^|x|) / 2 is not inf. Shifting the exponent as e^(|x| - ln(2))
  // avoids that overflow, but rounding error in |x| - ln(2) loses accuracy for
  // large |x|. Instead, we subtract c = 0x1.63p-1 (~= ln(2)), which has few
  // significant bits so |x| - c has no rounding error for 1 <= |x| < 1024, and
  // compensate with scale = e^c / 2 = 1 + scale_lo:
  //   sinh(|x|) = (e^|x| - e^-|x|) / 2
  //             = (e^c * e^(|x| - c) - e^c * e^(-c - |x|)) / 2
  //             = (e^c / 2) * (e^(|x| - c) - e^(-c - |x|))
  //             = scale * e^(|x| - c) - scale * e^(-c - |x|)
  //             = e^(|x| - c) + (e^(|x| - c) * scale_lo - e^(-c - |x|) * scale)
  auto c = ScalarLike(x, 0x1.63p-1);
  auto scale = ScalarLike(x, 0x1.000de868e1828p+0);      // e^c / 2
  auto scale_lo = ScalarLike(x, 0x1.bd0d1c3050e04p-13);  // e^c / 2 - 1
  XlaOp y = primitive_util::IsComplexType(type) ? x : Abs(x);
  auto exp_add = Exp(y - c, highest_accuracy);
  auto exp_sub = Exp(-c - y, highest_accuracy);
  auto large_sinh_result = exp_add + (exp_add * scale_lo - exp_sub * scale);
  if (primitive_util::IsComplexType(type)) {
    return large_sinh_result;
  }

  // For float64 and |x| < 1, avoid cancellation in e^|x| - e^-|x| near 0 by
  // writing sinh(|x|) in terms of t = expm1(|x|) = e^|x| - 1 (so e^|x| = 1 +
  // t):
  //   sinh(|x|) = (e^|x| - e^-|x|) / 2
  //             = (e^|x| - 1 / e^|x|) / 2
  //             = ((1 + t) - 1 / (1 + t)) / 2
  //             = ((1 + t)^2 - 1) / (2 * (1 + t))
  //             = (2t + t^2) / (2 * (1 + t))
  //             = (2t * (1 + t) - t^2) / (2 * (1 + t))
  //             = t - t^2 / (2 * (1 + t))
  // Keeping t as the leading term and applying the division only to the O(t^2)
  // correction damps rounding errors from 1 + t and the division by a factor of
  // t / (2 * (1 + t)).
  auto one_half = ScalarLike(x, 0.5);
  auto one = ScalarLike(y, 1.0);
  auto t = Expm1(y, highest_accuracy);
  auto small_sinh_result = t - one_half * (t * t) / (one + t);
  auto result = Select(Lt(y, one), small_sinh_result, large_sinh_result);
  return Sign(x) * result;
}

}  // namespace

bool ElementaryFunctionExpander::InstructionMatchesPattern(
    HloInstruction* instruction) {
  switch (instruction->opcode()) {
    case HloOpcode::kSinh: {
      PrimitiveType type = instruction->shape().element_type();
      return type == F64 || type == C64 || type == C128;
    }
    default:
      return false;
  }
}

absl::StatusOr<HloInstruction*> ElementaryFunctionExpander::ExpandInstruction(
    HloInstruction* instruction) {
  XlaBuilder b(std::string(instruction->name()));
  XlaOp x = Parameter(&b, 0, instruction->operand(0)->shape(), "x");
  switch (instruction->opcode()) {
    case HloOpcode::kSinh:
      ExpandSinh(x, instruction->shape().element_type());
      break;
    default:
      return absl::InternalError(
          absl::StrCat("Unsupported instruction: ", instruction->ToString()));
  }
  ABSL_ASSIGN_OR_RETURN(XlaComputation xla_computation, b.Build());
  ABSL_ASSIGN_OR_RETURN(HloComputation * computation,
                   XlaComputationToHloComputation(xla_computation,
                                                  instruction->GetModule()));
  HloInstruction* call =
      instruction->parent()->AddInstruction(HloInstruction::CreateCall(
          instruction->shape(), instruction->operands(), computation));
  ABSL_ASSIGN_OR_RETURN(auto inlined_map, CallInliner::Inline(call));
  HloInstruction* new_root = inlined_map[computation->root_instruction()];
  ABSL_RETURN_IF_ERROR(
      instruction->GetModule()->RemoveEmbeddedComputation(computation));
  return new_root;
}

}  // namespace xla
