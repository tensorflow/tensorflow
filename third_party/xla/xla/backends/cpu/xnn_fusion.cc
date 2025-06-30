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
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/dot_lib.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
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

absl::StatusOr<bool> IsDotSupportedByXnn(
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

const absl::flat_hash_map<HloOpcode, xnn_unary_operator>* GetXnnUnaryOpMap() {
  // TODO(ashaposhnikov): Investigate adding support for kErf, kExpm1, kLog1p,
  // kNot, kRoundNearestAfz, kTan.
  static const auto* map =
      new absl::flat_hash_map<HloOpcode, xnn_unary_operator>{
          {HloOpcode::kAbs, xnn_unary_abs},
          {HloOpcode::kCeil, xnn_unary_ceiling},
          {HloOpcode::kClz, xnn_unary_count_leading_zeros},
          {HloOpcode::kConvert, xnn_unary_convert},
          {HloOpcode::kCos, xnn_unary_cosine},
          {HloOpcode::kExp, xnn_unary_exp},
          {HloOpcode::kCbrt, xnn_unary_cube_root},
          {HloOpcode::kFloor, xnn_unary_floor},
          {HloOpcode::kLog, xnn_unary_log},
          {HloOpcode::kLogistic, xnn_unary_sigmoid},
          {HloOpcode::kNegate, xnn_unary_negate},
          {HloOpcode::kRoundNearestEven, xnn_unary_bankers_rounding},
          {HloOpcode::kRsqrt, xnn_unary_reciprocal_square_root},
          {HloOpcode::kSign, xnn_unary_sign},
          {HloOpcode::kSin, xnn_unary_sine},
          {HloOpcode::kSqrt, xnn_unary_square_root},
          {HloOpcode::kTanh, xnn_unary_tanh}};
  return map;
}

absl::StatusOr<xnn_unary_operator> XnnUnaryOperator(const HloOpcode& opcode) {
  const absl::flat_hash_map<HloOpcode, xnn_unary_operator>* map =
      GetXnnUnaryOpMap();
  auto result = map->find(opcode);
  if (result == map->end()) {
    return InvalidArgument("Unsupported XNNPACK unary operator: %s",
                           HloOpcodeString(opcode));
  }
  return result->second;
}

const absl::flat_hash_map<HloOpcode, xnn_binary_operator>* GetXnnBinaryOpMap() {
  static const auto* map =
      new absl::flat_hash_map<HloOpcode, xnn_binary_operator>{
          {HloOpcode::kAdd, xnn_binary_add},
          {HloOpcode::kAnd, xnn_binary_bitwise_and},
          {HloOpcode::kDivide, xnn_binary_divide},
          {HloOpcode::kMaximum, xnn_binary_maximum},
          {HloOpcode::kMinimum, xnn_binary_minimum},
          {HloOpcode::kMultiply, xnn_binary_multiply},
          {HloOpcode::kOr, xnn_binary_bitwise_or},
          {HloOpcode::kPower, xnn_binary_pow},
          {HloOpcode::kRemainder, xnn_binary_modulus},
          {HloOpcode::kShiftLeft, xnn_binary_shift_left},
          {HloOpcode::kShiftRightArithmetic, xnn_binary_shift_right_arithmetic},
          {HloOpcode::kShiftRightLogical, xnn_binary_shift_right_logical},
          {HloOpcode::kSubtract, xnn_binary_subtract},
          {HloOpcode::kXor, xnn_binary_bitwise_xor}};
  return map;
}

absl::StatusOr<xnn_binary_operator> XnnBinaryOperator(const HloOpcode& opcode) {
  const absl::flat_hash_map<HloOpcode, xnn_binary_operator>* map =
      GetXnnBinaryOpMap();
  auto result = map->find(opcode);
  if (result == map->end()) {
    return InvalidArgument("Unsupported XNNPACK binary operator: %s",
                           HloOpcodeString(opcode));
  }
  return result->second;
}

bool IsLayoutSupportedByXnn(const Shape& shape) {
  return !shape.has_layout() || LayoutUtil::HasDescendingLayout(shape.layout());
}

bool IsConstantSupportedByXnn(const HloInstruction* hlo) {
  CHECK(hlo->IsConstant());

  if (!XnnDatatype(hlo->shape().element_type()).ok()) {
    return false;
  }

  return hlo->shape().IsArray();
}

bool IsElementwiseOpSupportedByXnn(const HloInstruction* hlo) {
  CHECK(hlo->IsElementwise());
  // In XLA IsElementwise is true for constants.
  CHECK(!hlo->IsConstant());

  if (!XnnDatatype(hlo->shape().element_type()).ok()) {
    return false;
  }

  switch (hlo->operand_count()) {
    case 1:
      return XnnUnaryOperator(hlo->opcode()).ok();
    case 2:
      return XnnBinaryOperator(hlo->opcode()).ok();
    default:
      return false;
  }
}

bool IsBitcastOpSupportedByXnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kBitcast);
  if (!XnnDatatype(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  return hlo->shape().element_type() == input->shape().element_type();
}

bool IsBroadcastOpSupportedByXnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kBroadcast);
  if (!XnnDatatype(hlo->shape().element_type()).ok()) {
    return false;
  }
  const absl::Span<const int64_t> dims =
      Cast<HloBroadcastInstruction>(hlo)->dimensions();
  if (dims.empty()) {
    return true;
  }
  if (!std::is_sorted(dims.begin(), dims.end())) {
    return false;
  }
  // TODO(ashaposhnikov): this case works well, but we should investigate the
  // performance regressions that occur if this condition is removed.
  return dims.back() + 1 == dims.size();
}

}  // namespace xla::cpu
