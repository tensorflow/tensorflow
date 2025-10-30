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
#include <tuple>

#include "ynnpack/include/ynnpack.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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

absl::StatusOr<bool> IsDotSupportedByYnn(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape) {
  // Stores tuple of allowed (input, output) dtypes.
  static const absl::NoDestructor<absl::flat_hash_set<
      std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>>>
      kAllowedTypes({
          // TODO(b/452693819): We plan to enable this in stages, starting with
          // bf16 and int8, and enable f32 later.
          // {F32, F32, F32},
          // TODO(b/449998002): We don't have fast fp16 kernels yet.
          // {F16, F16, F32},
          {BF16, BF16, F32},
          {S8, S8, S32},
          {U8, S8, S32},
          // TODO(b/441600372): We don't have fast int4 kernels yet. Even the
          // reference kernel might be pretty good though?
          // {S8, S4, S32},
      });

  // Types must be in the allowed set.
  PrimitiveType lhs_dtype = lhs_shape.element_type();
  PrimitiveType rhs_dtype = rhs_shape.element_type();
  PrimitiveType out_dtype = out_shape.element_type();
  if (!kAllowedTypes->contains({lhs_dtype, rhs_dtype, out_dtype})) {
    return false;
  }

  if (!IsLayoutSupportedByYnn(lhs_shape) ||
      !IsLayoutSupportedByYnn(rhs_shape) ||
      !IsLayoutSupportedByYnn(out_shape)) {
    return false;
  }

  // Check shapes.
  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  if (dot_canonical_dims.m == 1 && dot_canonical_dims.n == 1 &&
      dot_shape.batch_size > 1) {
    // TODO(b/430079105): YNNPACK does not handle batch dimensions that are not
    // matrix dimensions. We could handle this case by fully implementing dot
    // (b/430079105), but we also could just insert dummy dimensions of size 1
    // for the matrix dimensions, so the batch dimensions get handled correctly.
    return false;
  }

  // YNNPACK supports transposing the inputs efficiently if possible (they will
  // fuse with dot packing), but we don't currently support generating the
  // necessary transposes.
  if (!dot_canonical_dims.lhs_canonical ||
      dot_canonical_dims.lhs_column_major ||
      dot_canonical_dims.rhs_column_major) {
    return false;
  }

  return true;
}

bool IsReduceOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kReduce);
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloReduceInstruction* reduce = Cast<HloReduceInstruction>(hlo);
  CHECK_NE(reduce, nullptr);
  // TODO(ashaposhnikov): we can support this edge case,
  // planning to come back to this later.
  if (reduce->dimensions().empty()) {
    return false;
  }

  HloInstruction* init = reduce->init_values().front();
  const PrimitiveType type = init->shape().element_type();
  // TODO(ashaposhnikov): The list of supported types can be extended.
  if (type != F32) {
    return false;
  }
  if (type != hlo->shape().element_type()) {
    return false;
  }

  const HloComputation* to_apply = reduce->to_apply();
  CHECK_NE(to_apply, nullptr);
  return Match(to_apply->root_instruction(),
               match::AnyOf<HloInstruction>(match::Add(), match::Maximum(),
                                            match::Minimum())
                   .WithBinaryOperandsAnyOrder(match::Parameter(0),
                                               match::Parameter(1)));
}

}  // namespace xla::cpu
