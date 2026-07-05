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
#include "absl/algorithm/container.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/cpu/runtime/dot_dims.h"
#include "xla/backends/cpu/runtime/ynnpack/ynn_interop.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
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
          {HloOpcode::kErf, ynn_unary_erf},
          {HloOpcode::kExp, ynn_unary_exp},
          {HloOpcode::kExpm1, ynn_unary_expm1},
          {HloOpcode::kFloor, ynn_unary_floor},
          {HloOpcode::kLog, ynn_unary_log},
          {HloOpcode::kLog1p, ynn_unary_log1p},
          {HloOpcode::kLogistic, ynn_unary_sigmoid},
          {HloOpcode::kNegate, ynn_unary_negate},
          {HloOpcode::kRoundNearestEven, ynn_unary_round},
          {HloOpcode::kRsqrt, ynn_unary_reciprocal_square_root},
          {HloOpcode::kSign, ynn_unary_sign},
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

absl::StatusOr<ynn_reduce_operator> YnnReduceOperator(const HloOpcode& opcode) {
  switch (opcode) {
    case HloOpcode::kAdd:
      return ynn_reduce_sum;
    case HloOpcode::kMaximum:
      return ynn_reduce_max;
    case HloOpcode::kMinimum:
      return ynn_reduce_min;
    default:
      return InvalidArgument("Unsupported YNNPACK reduce operator: %s",
                             HloOpcodeString(opcode));
  }
}

bool IsLayoutSupportedByYnn(const Shape& shape) {
  if (!shape.IsArray()) {
    return false;
  }
  if (shape.dimensions().empty()) {
    return true;
  }
  if (shape.dimensions().size() > YNN_MAX_TENSOR_RANK) {
    // TODO(b/460602165): We should eliminate this limitation.
    return false;
  }
  return !shape.has_layout() || LayoutUtil::HasDescendingLayout(shape.layout());
}

namespace {

bool CheckOperandCount(const HloInstruction* hlo, int num_operands) {
  if (hlo->operands().size() != num_operands) {
    return false;
  }
  return !absl::c_contains(hlo->operands(), nullptr);
}

bool HasAtLeastMinElements(const Shape& shape, int64_t min_elements) {
  return shape.IsArray() && ShapeUtil::ElementsIn(shape) >= min_elements;
}

}  // namespace

bool IsBitcastOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kBitcast);
  if (!CheckOperandCount(hlo, 1)) {
    return false;
  }
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  if (!IsLayoutSupportedByYnn(hlo->shape()) ||
      !IsLayoutSupportedByYnn(input->shape())) {
    return false;
  }

  return hlo->shape().element_type() == input->shape().element_type();
}

bool IsReshapeOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kReshape);
  if (!CheckOperandCount(hlo, 1)) {
    return false;
  }
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  if (hlo->shape().element_type() != input->shape().element_type()) {
    return false;
  }
  if (!IsLayoutSupportedByYnn(hlo->shape()) ||
      !IsLayoutSupportedByYnn(input->shape())) {
    return false;
  }

  return ShapeUtil::ReshapeIsBitcast(input->shape(), hlo->shape());
}

bool IsTransposeOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kTranspose);
  if (!CheckOperandCount(hlo, 1)) {
    return false;
  }
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  if (hlo->shape().element_type() != input->shape().element_type()) {
    return false;
  }
  return IsLayoutSupportedByYnn(hlo->shape()) &&
         IsLayoutSupportedByYnn(input->shape());
}

bool IsBroadcastOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kBroadcast);
  if (!CheckOperandCount(hlo, 1)) {
    return false;
  }
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  if (!IsLayoutSupportedByYnn(hlo->shape()) ||
      !IsLayoutSupportedByYnn(input->shape())) {
    return false;
  }

  // YNNPACK's broadcast operation can insert new dimensions, but not transpose.
  // HLO broadcast is more general. For now, let's only support "simple"
  // broadcasts that can be achieved by reshape + broadcast in YNNPACK. A
  // broadcast is "simple" if it preserves the relative order of operand
  // dimensions.
  auto dimensions = hlo->dimensions();
  for (int i = 1; i < dimensions.size(); ++i) {
    if (dimensions[i] <= dimensions[i - 1]) {
      return false;
    }
  }
  return true;
}

bool IsConcatenateOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kConcatenate);
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  if (!IsLayoutSupportedByYnn(hlo->shape())) {
    return false;
  }
  for (const HloInstruction* operand : hlo->operands()) {
    if (!operand) {
      return false;
    }
    if (hlo->shape().element_type() != operand->shape().element_type()) {
      return false;
    }
    if (!IsLayoutSupportedByYnn(operand->shape())) {
      return false;
    }
  }
  return true;
}

bool IsSliceOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kSlice);
  if (!CheckOperandCount(hlo, 1)) {
    return false;
  }
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  if (!IsLayoutSupportedByYnn(hlo->shape()) ||
      !IsLayoutSupportedByYnn(input->shape())) {
    return false;
  }

  return hlo->shape().element_type() == input->shape().element_type();
}

bool IsPadOpSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kPad);
  if (!CheckOperandCount(hlo, 2)) {
    return false;
  }
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  const HloInstruction* padding_value = hlo->operand(1);
  if (hlo->shape().element_type() != input->shape().element_type() ||
      hlo->shape().element_type() != padding_value->shape().element_type()) {
    return false;
  }
  if (!IsLayoutSupportedByYnn(hlo->shape()) ||
      !IsLayoutSupportedByYnn(input->shape())) {
    return false;
  }

  const PaddingConfig& config = hlo->padding_config();
  for (int i = 0; i < config.dimensions().size(); ++i) {
    const auto& dim = config.dimensions(i);
    if (input->shape().dimensions()[i] == 1) {
      if (dim.edge_padding_low() != 0 || dim.edge_padding_high() != 0) {
        // YNNPACK treats extent 1 dimensions as broadcasts (b/510492094).
        return false;
      }
    }
    if (dim.interior_padding() != 0) {
      // YNNPACK's ynn_define_static_pad does not support interior padding.
      return false;
    }
  }
  return true;
}

bool IsIotaSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kIota);
  PrimitiveType type = hlo->shape().element_type();
  if (type != F32 && type != S32) {
    return false;
  }
  return IsLayoutSupportedByYnn(hlo->shape());
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

  // Stores tuple of allowed dtypes.
  static const absl::NoDestructor<absl::flat_hash_set<PrimitiveType>>
      kAllowedTypes({
          F64,
          F32,
          BF16,
          S8,
          U8,
      });

  if (!kAllowedTypes->contains(hlo->shape().element_type())) {
    return false;
  }

  if (absl::c_any_of(hlo->operands(), [](const HloInstruction* op) {
        return !kAllowedTypes->contains(op->shape().element_type());
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

absl::StatusOr<bool> IsDotSupportedByYnn(const HloInstruction* hlo) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kDot);
  if (!CheckOperandCount(hlo, 2)) {
    return false;
  }
  const DotDimensionNumbers& dot_dimensions =
      Cast<HloDotInstruction>(hlo)->dot_dimension_numbers();
  const HloInstruction* lhs = hlo->operand(0);
  const HloInstruction* rhs = hlo->operand(1);
  const Shape& lhs_shape = lhs->shape();
  const Shape& rhs_shape = rhs->shape();
  const Shape& out_shape = hlo->shape();

  // Stores tuple of allowed (input, output) dtypes.
  static const absl::NoDestructor<absl::flat_hash_set<
      std::tuple<PrimitiveType, PrimitiveType, PrimitiveType>>>
      kAllowedTypes({
          {F64, F64, F64},
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
  ASSIGN_OR_RETURN(DotShape dot_shape, GetDotShape(dot_dimensions, lhs_shape,
                                                   rhs_shape, out_shape));

  ASSIGN_OR_RETURN(DotCanonicalDims dot_canonical_dims,
                   GetDotCanonicalDims(dot_dimensions, dot_shape));

  if (dot_canonical_dims.m == 1 || dot_canonical_dims.n == 1) {
    // TODO(b/430079105): YNNPACK does not handle vectors in dots. We could
    // insert dummy extent 1 dimensions to handle this case. We don't expect to
    // see this because XLA handles these with its own codegen, but if dots get
    // pulled into fusions, we need to reject them (b/469236467).
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

bool IsReduceLikeOpSupportedByYnn(const HloInstruction* hlo) {
  if (!CheckOperandCount(hlo, 2)) {
    return false;
  }
  if (!YnnType(hlo->shape().element_type()).ok()) {
    return false;
  }
  const HloInstruction* input = hlo->operand(0);
  if (!IsLayoutSupportedByYnn(hlo->shape()) ||
      !IsLayoutSupportedByYnn(input->shape())) {
    return false;
  }

  PrimitiveType input_dtype = input->shape().element_type();
  PrimitiveType out_dtype = hlo->shape().element_type();

  const HloInstruction* init = nullptr;
  if (hlo->opcode() == HloOpcode::kReduce) {
    const HloReduceInstruction* reduce = Cast<HloReduceInstruction>(hlo);
    init = reduce->init_values().front();
    // TODO(ashaposhnikov): we can support this edge case,
    // planning to come back to this later.
    if (reduce->dimensions().empty()) {
      return false;
    }
  } else if (hlo->opcode() == HloOpcode::kReduceWindow) {
    const HloReduceWindowInstruction* reduce_window =
        Cast<HloReduceWindowInstruction>(hlo);
    init = reduce_window->init_values().front();
    const Window& window = reduce_window->window();
    int new_axis_count = 0;
    for (const WindowDimension& dim : window.dimensions()) {
      if (dim.size() > 1 || dim.stride() > 1) {
        // TODO(ashaposhnikov): consider relaxing the constraints below.
        if (dim.size() > 1 && dim.stride() != dim.size()) {
          // When a reduce-window has a stride greater than 1 on a dimension
          // with size 1, it effectively skips input elements, resulting in a
          // smaller output dimension.
          return false;
        }
        if (dim.base_dilation() != 1) {
          return false;
        }
        if (dim.window_dilation() != 1) {
          return false;
        }
        if (dim.window_reversal()) {
          return false;
        }
        new_axis_count++;
      }
    }

    // We do not currently handle the case where reduce->dimensions() is empty,
    // see the check above.
    if (new_axis_count == 0) {
      return false;
    }

    // The ReduceWindow operation is implemented by expanding the input tensor
    // with window dimensions. We need to make sure that the resulting tensor
    // rank does not exceed YNNPACK limit.
    if (reduce_window->shape().dimensions().size() + new_axis_count >
        YNN_MAX_TENSOR_RANK) {
      return false;
    }
  } else {
    return false;
  }

  if (init->shape().element_type() != out_dtype) {
    return false;
  }

  const HloComputation* to_apply = hlo->to_apply();
  CHECK_NE(to_apply, nullptr);
  if (Match(to_apply->root_instruction(),
            match::AnyOf<HloInstruction>(match::Add())
                .WithBinaryOperandsAnyOrder(match::Parameter(0),
                                            match::Parameter(1)))) {
    static const absl::NoDestructor<
        absl::flat_hash_set<std::tuple<PrimitiveType, PrimitiveType>>>
        kAllowedTypes({
            {F64, F64},
            {F32, F32},
            {F16, F32},
            {BF16, F32},
            // YNNPACK accumulates these with an F32 accumulator, which seems to
            // be invalid for some tests, e.g. sequence_layers/jax:dense_test
            // {BF16, BF16},
            // {F16, F16},
            {S8, S32},
            {U8, S32},
            // TODO(b/441600372): We don't have fast int4 kernels yet. Even the
            // reference kernel might be pretty good though?
            // {S4, S32},
        });
    return kAllowedTypes->contains({input_dtype, out_dtype});
  }
  if (Match(to_apply->root_instruction(),
            match::AnyOf<HloInstruction>(match::Maximum(), match::Minimum())
                .WithBinaryOperandsAnyOrder(match::Parameter(0),
                                            match::Parameter(1)))) {
    if (input_dtype != out_dtype) {
      return false;
    }
    static const absl::NoDestructor<absl::flat_hash_set<PrimitiveType>>
        kAllowedTypes({U8, S8, BF16, F16, F32, F64});
    return kAllowedTypes->contains(out_dtype);
  }
  return false;
}

bool IsConvolutionOpSupportedByYnn(const HloInstruction* instr) {
  CHECK_EQ(instr->opcode(), HloOpcode::kConvolution);
  if (!CheckOperandCount(instr, 2)) {
    return false;
  }
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
      kAllowedTypes({
          {F64, F64, F64},
          {F32, F32, F32},
          // {BF16, BF16, F32},
          {S8, S8, S32},
      });

  const Shape& lhs_shape = conv->operand(0)->shape();
  const Shape& rhs_shape = conv->operand(1)->shape();
  const Shape& out_shape = conv->shape();

  if (!IsLayoutSupportedByYnn(lhs_shape) ||
      !IsLayoutSupportedByYnn(rhs_shape) ||
      !IsLayoutSupportedByYnn(out_shape)) {
    return false;
  }

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

bool IsInstructionPreferredByYnn(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kDot ||
      instr->opcode() == HloOpcode::kConvolution) {
    // IsDotSupportedByYnn has its own check here.
    return true;
  }
  constexpr int64_t kMinElements = 4096;

  if (HasAtLeastMinElements(instr->shape(), kMinElements)) {
    return true;
  }
  for (const HloInstruction* operand : instr->operands()) {
    if (HasAtLeastMinElements(operand->shape(), kMinElements)) {
      return true;
    }
  }
  return false;
}

uint32_t YnnFlags(const DebugOptions& debug_options) {
  uint32_t flags = 0;
  if (!debug_options.xla_cpu_enable_platform_dependent_math()) {
    flags |= YNN_FLAG_CONSISTENT_ARITHMETIC;
  }
  if (!debug_options.xla_allow_excess_precision()) {
    flags |= YNN_FLAG_NO_EXCESS_PRECISION;
  }
  return flags;
}

}  // namespace xla::cpu
