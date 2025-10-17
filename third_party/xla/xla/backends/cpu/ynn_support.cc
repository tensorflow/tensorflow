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

#include "xla/backends/cpu/ynn_support.h"

#include <algorithm>

#include "ynnpack/include/ynnpack.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/util.h"

namespace xla::cpu {

const absl::flat_hash_map<HloOpcode, ynn_unary_operator>& GetYnnUnaryOpMap() {
  static absl::NoDestructor<absl::flat_hash_map<HloOpcode, ynn_unary_operator>>
      unary_op_map({
          {HloOpcode::kAbs, ynn_unary_abs},
          {HloOpcode::kCeil, ynn_unary_ceil},
          {HloOpcode::kConvert, ynn_unary_convert},
          {HloOpcode::kCos, ynn_unary_cosine},
          {HloOpcode::kExp, ynn_unary_exp},
          {HloOpcode::kCbrt, ynn_unary_cube_root},
          {HloOpcode::kFloor, ynn_unary_floor},
          {HloOpcode::kLog, ynn_unary_log},
          {HloOpcode::kLogistic, ynn_unary_sigmoid},
          {HloOpcode::kNegate, ynn_unary_negate},
          {HloOpcode::kRoundNearestEven, ynn_unary_round},
          {HloOpcode::kRsqrt, ynn_unary_reciprocal_square_root},
          {HloOpcode::kSign, ynn_unary_sign},
          {HloOpcode::kSin, ynn_unary_sine},
          {HloOpcode::kSqrt, ynn_unary_square_root},
          {HloOpcode::kTanh, ynn_unary_tanh},
      });
  return *unary_op_map;
}

absl::StatusOr<ynn_unary_operator> YnnUnaryOperator(const HloOpcode& opcode) {
  const auto& unary_op_map = GetYnnUnaryOpMap();
  auto result = unary_op_map.find(opcode);
  if (result == unary_op_map.end()) {
    return InvalidArgument("Unsupported YNNPACK unary operator: %s",
                           HloOpcodeString(opcode));
  }
  return result->second;
}

const absl::flat_hash_map<HloOpcode, ynn_binary_operator>& GetYnnBinaryOpMap() {
  static absl::NoDestructor<absl::flat_hash_map<HloOpcode, ynn_binary_operator>>
      binary_op_map({
          {HloOpcode::kAdd, ynn_binary_add},
          {HloOpcode::kDivide, ynn_binary_divide},
          {HloOpcode::kMaximum, ynn_binary_max},
          {HloOpcode::kMinimum, ynn_binary_min},
          {HloOpcode::kMultiply, ynn_binary_multiply},
          {HloOpcode::kPower, ynn_binary_pow},
          {HloOpcode::kSubtract, ynn_binary_subtract},
      });
  return *binary_op_map;
}

absl::StatusOr<ynn_binary_operator> YnnBinaryOperator(const HloOpcode& opcode) {
  const auto& binary_op_map = GetYnnBinaryOpMap();
  auto result = binary_op_map.find(opcode);
  if (result == binary_op_map.end()) {
    return InvalidArgument("Unsupported YNNPACK binary operator: %s",
                           HloOpcodeString(opcode));
  }
  return result->second;
}

bool IsLayoutSupportedByYnn(const Shape& shape) {
  return !shape.has_layout() || LayoutUtil::HasDescendingLayout(shape.layout());
}

bool IsBitcastOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kBitcast);
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  return hlo->shape().element_type() == input->shape().element_type();
}

bool IsConstantSupportedByYnn(const HloInstruction* hlo) {
  CHECK(hlo->IsConstant());

  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }

  return hlo->shape().IsArray();
}

bool IsElementwiseOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK(hlo->IsElementwise());
  // In XLA IsElementwise is true for constants.
  CHECK(!hlo->IsConstant());

  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }

  if (!std::all_of(hlo->operands().begin(), hlo->operands().end(),
                   [](const HloInstruction* op) {
                     return YnnType(op->shape().element_type()).ok();
                   })) {
    return false;
  }

  switch (hlo->operand_count()) {
    case 1:
      return YnnUnaryOperator(hlo->opcode()).ok();
    case 2:
      return YnnBinaryOperator(hlo->opcode()).ok();
    default:
      return false;
  }
}

}  // namespace xla::cpu
