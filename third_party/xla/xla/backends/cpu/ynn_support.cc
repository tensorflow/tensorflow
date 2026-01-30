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
#include <cstdint>
#include <tuple>

#include "ynnpack/include/ynnpack.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/dot_dims.h"
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
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

const absl::flat_hash_map<HloOpcode, ynn_unary_operator>& GetYnnUnaryOpMap() {
  static absl::NoDestructor<absl::flat_hash_map<HloOpcode, ynn_unary_operator>>
      unary_op_map({
          {HloOpcode::kAbs, ynn_unary_abs},
          {HloOpcode::kCeil, ynn_unary_ceil},
          {HloOpcode::kConvert, ynn_unary_convert},
          {HloOpcode::kCos, ynn_unary_cosine},
          {HloOpcode::kErf, ynn_unary_erf},
          {HloOpcode::kExp, ynn_unary_exp},
          {HloOpcode::kExpm1, ynn_unary_expm1},
          {HloOpcode::kCbrt, ynn_unary_cube_root},
          {HloOpcode::kFloor, ynn_unary_floor},
          {HloOpcode::kLog, ynn_unary_log},
          {HloOpcode::kLog1p, ynn_unary_log1p},
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
  if (shape.dimensions().size() > YNN_MAX_TENSOR_RANK) {
    // TODO(b/460602165): We should eliminate this limitation.
    return false;
  }
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

  // We don't want to handle ops that are too small, overhead will be
  // significant.
  // TODO(b/469236467): This threshold is probably too small in some cases and
  // too big in others.
  constexpr int64_t kMinElements = 64;
  if (ShapeUtil::ElementsIn(hlo->shape()) < kMinElements) {
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
          {F32, F32, F32},
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

  if ((dot_canonical_dims.m == 1 || dot_canonical_dims.n == 1) &&
      dot_shape.batch_size > 1) {
    // TODO(b/430079105): YNNPACK does not handle batch dimensions that are not
    // matrix dimensions. We could handle this case by fully implementing dot
    // (b/430079105), but we also could just insert dummy dimensions of size 1
    // for the matrix dimensions, so the batch dimensions get handled correctly.
    return false;
  }

  if (std::max({dot_canonical_dims.m, dot_canonical_dims.k,
                dot_canonical_dims.n}) < 8) {
    // If this dot is small, our overhead is probably too significant.
    // TODO(b/458529782): This is here as a workaround for an unrelated bug.
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

absl::StatusOr<bool> IsDotSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kDot);
  return IsDotSupportedByYnn(hlo->dot_dimension_numbers(),
                             hlo->operand(0)->shape(), hlo->operand(1)->shape(),
                             hlo->shape());
}

bool IsReduceOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kReduce);
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  if (!IsLayoutSupportedByYnn(hlo->shape()) ||
      !IsLayoutSupportedByYnn(hlo->operand(0)->shape())) {
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

bool IsReduceOpOffloadedToYnn(const HloInstruction* hlo) {
  if (!IsReduceOpSupportedByYnn(hlo)) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  if (ShapeUtil::ElementsIn(input->shape()) < 32 * 1024) {
    return false;
  }
  switch (input->opcode()) {
    case HloOpcode::kMultiply:
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kSlice:
    case HloOpcode::kConcatenate:
    case HloOpcode::kConvert:
    case HloOpcode::kReshape:
      return false;
    default: {
      return true;
    }
  }
}

bool IsConvolutionOpSupportedByYnn(const HloInstruction* instr) {
  CHECK_EQ(instr->opcode(), HloOpcode::kConvolution);
  const HloConvolutionInstruction* conv =
      Cast<HloConvolutionInstruction>(instr);

  ConvolutionDimensionNumbers conv_dimensions =
      conv->convolution_dimension_numbers();
  Window window = conv->window();

  if (conv->batch_group_count() != 1) {
    return false;
  }

  // Only support 2D convolution.
  if (window.dimensions_size() != 2) {
    return false;
  }

  // Stores tuple of allowed (input, output) dtypes.
  // TODO(b/466474339): Enable other data types.
  static const absl::NoDestructor<absl::flat_hash_set<
      std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>>>
      kAllowedTypes({/*{F32, F32, F32}, {BF16, BF16, F32},*/ {S8, S8, S32}});

  const Shape& lhs_shape = conv->operand(0)->shape();
  const Shape& rhs_shape = conv->operand(1)->shape();
  const Shape& out_shape = conv->shape();

  PrimitiveType lhs_dtype = lhs_shape.element_type();
  PrimitiveType rhs_dtype = rhs_shape.element_type();
  PrimitiveType out_dtype = out_shape.element_type();

  if (!kAllowedTypes->contains({lhs_dtype, rhs_dtype, out_dtype})) {
    return false;
  }

  // Make sure that this layout is supported.
  if (conv_dimensions.input_feature_dimension() != 3 ||
      conv_dimensions.output_feature_dimension() != 3) {
    return false;
  }

  if (conv_dimensions.kernel_input_feature_dimension() != 2 ||
      conv_dimensions.kernel_output_feature_dimension() != 3) {
    return false;
  }

  if (conv_dimensions.input_spatial_dimensions_size() != 2 ||
      conv_dimensions.kernel_spatial_dimensions_size() != 2 ||
      conv_dimensions.output_spatial_dimensions_size() != 2) {
    return false;
  }

  if (conv_dimensions.input_spatial_dimensions(0) != 1 ||
      conv_dimensions.input_spatial_dimensions(1) != 2 ||
      conv_dimensions.kernel_spatial_dimensions(0) != 0 ||
      conv_dimensions.kernel_spatial_dimensions(1) != 1 ||
      conv_dimensions.output_spatial_dimensions(0) != 1 ||
      conv_dimensions.output_spatial_dimensions(1) != 2) {
    return false;
  }

  if (std::max({
          lhs_shape.dimensions(conv_dimensions.input_feature_dimension()),
          out_shape.dimensions(conv_dimensions.output_feature_dimension()),
      }) <= 16) {
    // If this  convolution is small, our overhead is probably too significant.
    // TODO(b/458529782, b/473570788): This is here as a workaround for an
    // unrelated bug.
    return false;
  }

  // Skip if output or filter is larger than input.
  // TODO(b/476207717): this should work fine in theory, but currently this
  // fails at one of the shape checks fails as statically false. I think the
  // issue is that an inferred input size is larger than what was provided.
  for (int i = 0; i < conv_dimensions.input_spatial_dimensions_size(); ++i) {
    if (out_shape.dimensions(conv_dimensions.output_spatial_dimensions(i)) >
        lhs_shape.dimensions(conv_dimensions.input_spatial_dimensions(i))) {
      return false;
    }
    if (rhs_shape.dimensions(conv_dimensions.kernel_spatial_dimensions(i)) >
        lhs_shape.dimensions(conv_dimensions.input_spatial_dimensions(i))) {
      return false;
    }
  }

  // No base dilation for now.
  if ((window.dimensions(0).base_dilation() != 1) ||
      (window.dimensions(1).base_dilation() != 1)) {
    return false;
  }

  // TODO(b/474103597): we might be able to do this using negative strides,
  // but this feature is rarely used and considered for deprecation.
  if ((window.dimensions(0).window_reversal() != 0) ||
      (window.dimensions(1).window_reversal() != 0)) {
    return false;
  }

  return true;
}

uint32_t YnnFlags(const DebugOptions& debug_options) {
  uint32_t flags = 0;
  if (!debug_options.xla_cpu_enable_platform_dependent_math()) {
    flags |= YNN_FLAG_CONSISTENT_ARITHMETIC;
  }
  return flags;
}

}  // namespace xla::cpu
