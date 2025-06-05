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

#include "xla/backends/cpu/xnn_fusion.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include "xnnpack.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

// Thresholds for when to use thread pool for XNNPACK fusions for different
// HLOs. These numbers picked up randomly and need benchmarks to tune.
static constexpr int64_t kDotThreshold = 10 * 1000;
static constexpr int64_t kDefaultThreshold = 100 * 1000;

static int64_t MaxElementsCount(const Shape& shape) {
  int64_t ret = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& shape, const ShapeIndex& index) {
        ret = std::max(ret, ShapeUtil::ElementsIn(shape));
      });
  return ret;
}

// We rely on a very simple heuristic to determine if thread pool is beneficial
// for XNNPACK fusions. We assume that if the HLO produces a large result (or
// has large operands), thread pool will be beneficial for running operation in
// parallel. For small operations, thread pool overheads are higher than the
// actual computation.
static int64_t MaxElementsCount(const HloInstruction* hlo,
                                bool include_operands = true) {
  int64_t ret = MaxElementsCount(hlo->shape());
  if (include_operands) {
    for (auto* operand : hlo->operands()) {
      ret = std::max(ret, MaxElementsCount(operand->shape()));
    }
  }
  return ret;
}

bool XnnShouldUseThreadPool(const HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kDot:
      return MaxElementsCount(hlo) > kDotThreshold;
    default:
      return MaxElementsCount(hlo) > kDefaultThreshold;
  }
}

bool XnnShouldUseThreadPool(const HloComputation* computation) {
  return absl::c_any_of(
      computation->instructions(),
      [](const HloInstruction* hlo) { return XnnShouldUseThreadPool(hlo); });
}

bool AreDtypesSupported(const Shape& lhs_shape, const Shape& rhs_shape,
                        const Shape& out_shape,
                        const TargetMachineFeatures* cpu_features) {
  // Stores tuple of allowed (input, output) dtypes.
  static const auto* kAllowedTypes =
      new absl::flat_hash_set<std::pair<PrimitiveType, PrimitiveType>>(
          {{F32, F32}, {BF16, F32}, {BF16, BF16}});

  // Types must be in the allowed set.
  PrimitiveType lhs_dtype = lhs_shape.element_type();
  PrimitiveType rhs_dtype = rhs_shape.element_type();
  PrimitiveType out_dtype = out_shape.element_type();
  if (lhs_dtype != rhs_dtype ||
      !kAllowedTypes->contains({lhs_dtype, out_dtype})) {
    return false;
  }

  // BF16 matmuls can only run when CPU has AVX512_BF16.
  if (lhs_dtype == BF16) {
    return cpu_features == nullptr || cpu_features->has_avx512bf16();
  }
  return true;
}

absl::StatusOr<bool> IsXnnDotSupported(
    const DotDimensionNumbers& dot_dimensions, const Shape& lhs_shape,
    const Shape& rhs_shape, const Shape& out_shape,
    const TargetMachineFeatures* cpu_features) {
  // Check data types.
  if (!AreDtypesSupported(lhs_shape, rhs_shape, out_shape, cpu_features)) {
    return false;
  }

  // Check shapes.
  TF_ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                      rhs_shape, out_shape));

  TF_ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                      GetDotCanonicalDims(dot_dimensions, dot_shape));

  // TODO(b/385370486): XNNPACK does not tile by `K` and can be a lot slower
  // than the default Eigen implementation.
  if (dot_canonical_dims.k / dot_canonical_dims.m > 5 ||
      dot_canonical_dims.k / dot_canonical_dims.n > 5) {
    return false;
  }

  // XNNPACK does not support transposing LHS or col-major layouts.
  return dot_canonical_dims.lhs_canonical &&
         !dot_canonical_dims.lhs_column_major &&
         !dot_canonical_dims.rhs_column_major;
}

absl::StatusOr<xnn_datatype> XnnDatatype(const PrimitiveType& type) {
  switch (type) {
    case BF16:
      return xnn_datatype_bf16;
    case F16:
      return xnn_datatype_fp16;
    case F32:
      return xnn_datatype_fp32;
    default:
      return InvalidArgument("Unsupported XNNPACK data type: %s",
                             primitive_util::LowercasePrimitiveTypeName(type));
  }
}

absl::StatusOr<xnn_unary_operator> XnnUnaryOperator(const HloOpcode& opcode) {
  switch (opcode) {
    // TODO(ashaposhnikov): Need to debug the crash with kAbs.
    // case HloOpcode::kAbs:
    //  return xnn_unary_abs;
    case HloOpcode::kCeil:
      return xnn_unary_ceiling;
    case HloOpcode::kClz:
      return xnn_unary_count_leading_zeros;
    case HloOpcode::kConvert:
      return xnn_unary_convert;
    case HloOpcode::kCos:
      return xnn_unary_cosine;
    // TODO(ashaposhnikov): Need to debug the crash with kExp.
    // case HloOpcode::kExp:
    //  return xnn_unary_exp;
    case HloOpcode::kCbrt:
      return xnn_unary_cube_root;
    // case HloOpcode::kErf:
    // case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
      return xnn_unary_floor;
    case HloOpcode::kLog:
      return xnn_unary_log;
    // case HloOpcode::kLog1p:
    case HloOpcode::kLogistic:
      return xnn_unary_sigmoid;
    // TODO(ashaposhnikov): Need to debug the crash with kNegate.
    // case HloOpcode::kNegate:
    //  return xnn_unary_negate;
    // case HloOpcode::kNot:
    // case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kRoundNearestEven:
      return xnn_unary_bankers_rounding;
    case HloOpcode::kRsqrt:
      return xnn_unary_reciprocal_square_root;
    case HloOpcode::kSign:
      return xnn_unary_sign;
    case HloOpcode::kSin:
      return xnn_unary_sine;
    // TODO(ashaposhnikov): Need to debug the crash with kSqrt.
    // case HloOpcode::kSqrt:
    //  return xnn_unary_square_root;
    // case HloOpcode::kTan:
    case HloOpcode::kTanh:
      return xnn_unary_tanh;
    default:
      return InvalidArgument("Unsupported XNNPACK unary operator: %s",
                             HloOpcodeString(opcode));
  }
}

absl::StatusOr<xnn_binary_operator> XnnBinaryOperator(const HloOpcode& opcode) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return xnn_binary_add;
    case HloOpcode::kAnd:
      return xnn_binary_bitwise_and;
    // TODO(ashaposhnikov): debug crashes with these instructions.
    // case HloOpcode::kDivide:
    //  return xnn_binary_divide;
    // case HloOpcode::kMaximum:
    //  return xnn_binary_maximum;
    case HloOpcode::kMinimum:
      return xnn_binary_minimum;
    case HloOpcode::kMultiply:
      return xnn_binary_multiply;
    case HloOpcode::kOr:
      return xnn_binary_bitwise_or;
    case HloOpcode::kPower:
      return xnn_binary_pow;
    case HloOpcode::kRemainder:
      return xnn_binary_modulus;
    case HloOpcode::kShiftLeft:
      return xnn_binary_shift_left;
    case HloOpcode::kShiftRightArithmetic:
      return xnn_binary_shift_right_arithmetic;
    case HloOpcode::kShiftRightLogical:
      return xnn_binary_shift_right_logical;
    case HloOpcode::kSubtract:
      return xnn_binary_subtract;
    case HloOpcode::kXor:
      return xnn_binary_bitwise_xor;
    default:
      return InvalidArgument("Unsupported XNNPACK binary operator: %s",
                             HloOpcodeString(opcode));
  }
}

}  // namespace xla::cpu
