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

#include "xla/hlo/transforms/dot_algorithm_rewriter.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

// Maps each kept non-contracting dimension of a reduced operand scale tensor
// to its destination dimension index in dot_shape, inserting a transpose only
// when the target output dimensions are not strictly increasing.
absl::StatusOr<HloInstruction*> BroadcastScaleToDotOutput(
    HloInstruction* scale, absl::Span<const int64_t> operand_batch_dims,
    absl::Span<const int64_t> operand_contracting_dims, int64_t operand_rank,
    int64_t non_contracting_out_offset, const Shape& dot_shape) {
  if (dot_shape.dimensions().empty()) {
    return scale;
  }
  if (scale->shape().dimensions().empty()) {
    return MakeBroadcastHlo(scale, {}, dot_shape);
  }

  std::vector<int64_t> target_out_dims;
  target_out_dims.reserve(scale->shape().dimensions().size());
  int64_t nc_index = 0;
  for (int64_t d = 0; d < operand_rank; ++d) {
    if (absl::c_linear_search(operand_contracting_dims, d)) {
      continue;
    }
    auto batch_it = absl::c_find(operand_batch_dims, d);
    if (batch_it != operand_batch_dims.end()) {
      target_out_dims.push_back(
          std::distance(operand_batch_dims.begin(), batch_it));
    } else {
      target_out_dims.push_back(non_contracting_out_offset + nc_index);
      ++nc_index;
    }
  }

  if (!absl::c_is_sorted(target_out_dims)) {
    std::vector<int64_t> perm(target_out_dims.size());
    absl::c_iota(perm, 0);
    absl::c_sort(perm, [&](int64_t a, int64_t b) {
      return target_out_dims[a] < target_out_dims[b];
    });
    ABSL_ASSIGN_OR_RETURN(scale, MakeTransposeHlo(scale, perm));
    absl::c_sort(target_out_dims);
  }
  return MakeBroadcastHlo(scale, target_out_dims, dot_shape);
}

}  // namespace

// Rewrites BF16 or F32 dots with ALG_DOT_BF16_BF16_FP8X3/4 into per channel
// power of two scaling, two slice F8E4M3FN decomposition, and 3 or 4 F8E4M3FN
// matmuls accumulated in FP32.
absl::StatusOr<bool> RewriteFp8xNDot(
    HloInstruction* dot, PrecisionConfig::Precision f8_dot_precision) {
  const PrecisionConfig::Algorithm algorithm =
      dot->precision_config().algorithm();
  if (algorithm != PrecisionConfig::ALG_DOT_BF16_BF16_FP8X3 &&
      algorithm != PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4) {
    return false;
  }

  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  const PrimitiveType lhs_type = lhs->shape().element_type();
  const PrimitiveType rhs_type = rhs->shape().element_type();
  if ((lhs_type != BF16 && lhs_type != F32) ||
      (rhs_type != BF16 && rhs_type != F32)) {
    return false;
  }
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions().empty() ||
      dnums.rhs_contracting_dimensions().empty()) {
    return false;
  }

  // IEEE 754 binary32 exponent layout: 8 exponent bits [30:23] with bias 127.
  static constexpr uint32_t kF32MantissaBits = 23;
  static constexpr uint32_t kF32ExponentBias = 127;
  static constexpr uint32_t kF32ExponentMask = 0xFFu << kF32MantissaBits;
  // Scale each channel's amax into F8E4M3FN's top power of two binade [2^7,
  // 2^8) = [128, 256). For amax with unbiased exponent E and F32 biased
  // exponent B = E + 127, the scale 2^(7 - E) has F32 biased exponent (7 - E) +
  // 127 = (2 * 127 + 7) - B = 261 - B.
  static constexpr uint32_t kF8E4M3FNTopBinadeExponent = 7;
  static constexpr uint32_t kNegScaleBiasedExponentSumBits =
      (2 * kF32ExponentBias + kF8E4M3FNTopBinadeExponent) << kF32MantissaBits;
  // Undoes the 2^7 centering shift on both LHS and RHS: 2^(-2 * 7) = 2^-14.
  static constexpr float kDescaleTwo7thShifts2PowNeg14 = 0x1p-14f;
  // Floor clamp for all zero channels: 2^-120 has F32 biased exponent
  // 127 - 120 = 7, the smallest B where 261 - B = 254 stays a finite F32.
  static constexpr float kZeroChannelClampMin2PowNeg120 = 0x1p-120f;
  // F8E4M3FN has 4 significand bits (1 implicit + 3 mantissa), so the rounding
  // residual sits 4 binades below the high slice. Shift up by 2^4 to quantize
  // the low slice, then scale cross terms by 2^-4 and 2^-8 when accumulating.
  static constexpr float kLowSliceUpShift2Pow4 = 0x1p4f;
  static constexpr float kCrossTermDownScale2PowNeg4 = 0x1p-4f;
  static constexpr float kLowLowTermDownScale2PowNeg8 = 0x1p-8f;

  HloComputation* comp = dot->parent();

  struct QuantizedOperand {
    HloInstruction* high;
    HloInstruction* low;
    HloInstruction* exponent;
  };

  auto quantize_operand = [&](HloInstruction* op,
                              absl::Span<const int64_t> contracting_dims)
      -> absl::StatusOr<QuantizedOperand> {
    std::vector<int64_t> reduce_dims(contracting_dims.begin(),
                                     contracting_dims.end());
    absl::c_sort(reduce_dims);
    std::vector<int64_t> kept_dims;
    const int64_t op_rank = op->shape().dimensions().size();
    for (int64_t d = 0; d < op_rank; ++d) {
      if (!absl::c_linear_search(reduce_dims, d)) {
        kept_dims.push_back(d);
      }
    }

    HloInstruction* op_f32 = MakeConvertToHlo(op, F32);
    ABSL_ASSIGN_OR_RETURN(HloInstruction * abs_op,
                     MakeUnaryHlo(HloOpcode::kAbs, op_f32));

    // Dynamic exponent shift S = floor(log2(amax)) - 7 centers max magnitude in
    // [128.0, 256.0).
    HloInstruction* zero_f32 = MakeR0ConstantHlo<float>(comp, 0.0f);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * max_op,
        MakeReduceHlo(abs_op, zero_f32, reduce_dims, HloOpcode::kMaximum));
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * max_clamped,
        MakeBinaryHlo(HloOpcode::kMaximum, max_op,
                      MakeScalarLike(max_op, kZeroChannelClampMin2PowNeg120)));

    // Extract IEEE 754 biased exponent field and construct exact power of two
    // scale 2^(-S) via single unsigned subtract (261 << 23) - exp_op.
    HloInstruction* bits_op = MakeBitcastConvertToHlo(max_clamped, U32);
    ABSL_ASSIGN_OR_RETURN(HloInstruction * exp_op,
                     MakeBinaryHlo(HloOpcode::kAnd, bits_op,
                                   MakeScalarLike(bits_op, kF32ExponentMask)));
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * neg_s_bits,
        MakeBinaryHlo(HloOpcode::kSubtract,
                      MakeScalarLike(exp_op, kNegScaleBiasedExponentSumBits),
                      exp_op));
    HloInstruction* scale_neg = MakeBitcastConvertToHlo(neg_s_bits, F32);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * scaled,
        MakeBinaryHlo(HloOpcode::kMultiply, op_f32,
                      MakeBroadcastHlo(scale_neg, kept_dims, op_f32->shape())));

    // High slice x_high: convert scaled value to F8E4M3FN using round to
    // nearest even, then convert back to F32 for residual subtraction.
    HloInstruction* high = MakeConvertToHlo(scaled, F8E4M3FN);
    HloInstruction* high_f32 = MakeConvertToHlo(high, F32);

    // Residual r = x_scaled - x_high cancels top bits exactly by Sterbenz
    // Lemma, leaving signed lower mantissa bits centered around zero.
    ABSL_ASSIGN_OR_RETURN(HloInstruction * residual_f32,
                     MakeBinaryHlo(HloOpcode::kSubtract, scaled, high_f32));

    // Low slice x_low: shift residual up by 4 binades (* 2^4) into normal
    // F8E4M3FN exponent range.
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * low_f32,
        MakeBinaryHlo(HloOpcode::kMultiply, residual_f32,
                      MakeScalarLike(residual_f32, kLowSliceUpShift2Pow4)));

    return QuantizedOperand{high, MakeConvertToHlo(low_f32, F8E4M3FN),
                            MakeBitcastConvertToHlo(exp_op, F32)};
  };

  ABSL_ASSIGN_OR_RETURN(QuantizedOperand lhs_quant,
                   quantize_operand(lhs, dnums.lhs_contracting_dimensions()));
  ABSL_ASSIGN_OR_RETURN(QuantizedOperand rhs_quant,
                   quantize_operand(rhs, dnums.rhs_contracting_dimensions()));
  auto [a_high, a_low, exp_a_f32] = lhs_quant;
  auto [b_high, b_low, exp_b_f32] = rhs_quant;

  PrecisionConfig f8_pc;
  f8_pc.add_operand_precision(f8_dot_precision);
  f8_pc.add_operand_precision(f8_dot_precision);
  // Keep the dot's layout: on GPU this runs after layout assignment.
  const Shape dot_shape_f32 = ShapeUtil::ChangeElementType(dot->shape(), F32);
  auto make_f8_dot = [&](HloInstruction* a, HloInstruction* b) {
    return comp->AddInstruction(
        HloInstruction::CreateDot(dot_shape_f32, a, b, dnums, f8_pc),
        &dot->metadata());
  };

  // Cartesian cross terms: A * B = (A_low * B_low) * 2^-8 + (A_low * B_high +
  // A_high * B_low) * 2^-4 + A_high * B_high.
  HloInstruction* p10 = make_f8_dot(a_low, b_high);
  HloInstruction* p01 = make_f8_dot(a_high, b_low);
  HloInstruction* p00 = make_f8_dot(a_high, b_high);

  // Scale first order cross terms (p10 + p01) by 2^-4 to undo the 2^4 low slice
  // shift.
  ABSL_ASSIGN_OR_RETURN(HloInstruction * p10_p01_raw,
                   MakeBinaryHlo(HloOpcode::kAdd, p10, p01));
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * low_sum,
      MakeBinaryHlo(HloOpcode::kMultiply, p10_p01_raw,
                    MakeScalarLike(p10_p01_raw, kCrossTermDownScale2PowNeg4)));

  // Accumulate from smallest magnitude terms to largest: p11 + p10 + p01 + p00.
  if (algorithm == PrecisionConfig::ALG_DOT_BF16_BF16_FP8X4) {
    // Scale second order residual term p11 by 2^-8 for FP8x4.
    HloInstruction* p11_raw = make_f8_dot(a_low, b_low);
    ABSL_ASSIGN_OR_RETURN(
        HloInstruction * p11,
        MakeBinaryHlo(HloOpcode::kMultiply, p11_raw,
                      MakeScalarLike(p11_raw, kLowLowTermDownScale2PowNeg8)));
    ABSL_ASSIGN_OR_RETURN(low_sum, MakeBinaryHlo(HloOpcode::kAdd, p11, low_sum));
  }
  ABSL_ASSIGN_OR_RETURN(HloInstruction * sum_f32,
                   MakeBinaryHlo(HloOpcode::kAdd, low_sum, p00));

  // Restore true output dynamic range by multiplying accumulated FP32 sum by
  // per channel descale factor 2^(S_A + S_B) = exp_a_f32 * (exp_b_f32 * 2^-14).
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * scale_b,
      MakeBinaryHlo(HloOpcode::kMultiply, exp_b_f32,
                    MakeScalarLike(exp_b_f32, kDescaleTwo7thShifts2PowNeg14)));
  const int64_t lhs_rank = lhs->shape().dimensions().size();
  const int64_t rhs_rank = rhs->shape().dimensions().size();
  const int64_t num_batch_dims = dnums.lhs_batch_dimensions_size();
  const int64_t num_lhs_nc =
      lhs_rank - num_batch_dims - dnums.lhs_contracting_dimensions_size();
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * s_a_out,
      BroadcastScaleToDotOutput(exp_a_f32, dnums.lhs_batch_dimensions(),
                                dnums.lhs_contracting_dimensions(), lhs_rank,
                                /*non_contracting_out_offset=*/num_batch_dims,
                                dot_shape_f32));
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * s_b_out,
      BroadcastScaleToDotOutput(
          scale_b, dnums.rhs_batch_dimensions(),
          dnums.rhs_contracting_dimensions(), rhs_rank,
          /*non_contracting_out_offset=*/num_batch_dims + num_lhs_nc,
          dot_shape_f32));
  ABSL_ASSIGN_OR_RETURN(HloInstruction * descaled_a,
                   MakeBinaryHlo(HloOpcode::kMultiply, sum_f32, s_a_out));
  ABSL_ASSIGN_OR_RETURN(HloInstruction * descaled_f32,
                   MakeBinaryHlo(HloOpcode::kMultiply, descaled_a, s_b_out));
  HloInstruction* final_out =
      dot->shape().element_type() == F32
          ? descaled_f32
          : MakeConvertToHlo(descaled_f32, dot->shape().element_type());
  ABSL_RETURN_IF_ERROR(comp->ReplaceInstruction(dot, final_out));
  return true;
}

}  // namespace xla
