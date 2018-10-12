/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_
#define TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/round.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace reference_ops {

inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& output_shape, float* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find max element value which we'll use to ensure numerical stability
    // taking advantage of the following equality:
    // exp(x[i])/sum(exp(x[i])) == exp(x[i]+C)/sum(exp(x[i]+C))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp((input_data[i * depth + c] - max) * params.beta);
    }

    // Compute result.
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] =
          std::exp((input_data[i * depth + c] - max) * params.beta) / sum;
    }
  }
}

inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const uint8* input_data,
                    const RuntimeShape& output_shape, uint8* output_data) {
  const int32 input_beta_multiplier = params.input_multiplier;
  const int32 input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    uint8 max_in_row = 0;
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    int32 fixed_sum_of_exps = sum_of_exps.raw();
    int headroom_plus_one =
        CountLeadingZeros(static_cast<uint32>(fixed_sum_of_exps));
    // This is the number of bits to the left of the binary point above 1.0.
    // Consider fixed_sum_of_exps=1.25.  In that case shifted_scale=0.8 and
    // no later adjustment will be needed.
    int num_bits_over_unit = kAccumulationIntegerBits - headroom_plus_one;
    int32 shifted_sum_minus_one = static_cast<int32>(
        (static_cast<uint32>(fixed_sum_of_exps) << headroom_plus_one) -
        (static_cast<uint32>(1) << 31));

    FixedPoint0 shifted_scale = gemmlowp::one_over_one_plus_x_for_x_in_0_1(
        FixedPoint0::FromRaw(shifted_sum_minus_one));

    for (int c = 0; c < depth; ++c) {
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        int32 unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);

        output_data[i * depth + c] = static_cast<uint8>(
            std::max(std::min(unsat_output, static_cast<int32>(255)),
                     static_cast<int32>(0)));

      } else {
        output_data[i * depth + c] = 0;
      }
    }
  }
}

// Performs softmax along the input of size (input_size * batch_size).
inline void Softmax(const float* in, const int input_size, const int batch_size,
                    const float beta, float* out) {
  //  TF_LITE_ASSERT(input_size > 0);

  // For each batch
  for (int b = 0; b < batch_size; b++) {
    // Find the max coeff.
    float max_coeff = in[0];
    for (int i = 1; i < input_size; i++) {
      if (in[i] > max_coeff) max_coeff = in[i];
    }

    // Compute the normalized sum of exps.
    float exp_sum = 0.0;
    for (int i = 0; i < input_size; i++) {
      out[i] = std::exp((in[i] - max_coeff) * beta);
      exp_sum += out[i];
    }

    // Divide by the sum of exps.
    float reciprocal_sum_exp = 1.f / exp_sum;
    for (int i = 0; i < input_size; i++) {
      out[i] *= reciprocal_sum_exp;
    }

    // Advance in and out pointers for the next batch.
    in += input_size;
    out += input_size;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_
