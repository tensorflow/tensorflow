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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_

#include <limits>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/op_macros.h"

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
      const float exp_c = std::exp((input_data[i * depth + c] - max) *
                                   static_cast<float>(params.beta));
      output_data[i * depth + c] = exp_c;
      sum += exp_c;
    }

    // Compute result.
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] = output_data[i * depth + c] / sum;
    }
  }
}

// Quantized softmax with int8_t/uint8_t input and int8_t/uint8_t/int16_t
// output.
template <typename InputT, typename OutputT>
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const InputT* input_data,
                    const RuntimeShape& output_shape, OutputT* output_data) {
  const int32_t input_beta_multiplier = params.input_multiplier;
  const int32_t input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
  using FixedPointAccum =
      gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    InputT max_in_row = std::numeric_limits<InputT>::min();
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        int32_t unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(),
            num_bits_over_unit + 31 - (sizeof(OutputT) * 8));

        const int32_t shifted_output =
            unsat_output +
            static_cast<int32_t>(std::numeric_limits<OutputT>::min());

        output_data[i * depth + c] = static_cast<OutputT>(std::max(
            std::min(shifted_output,
                     static_cast<int32_t>(std::numeric_limits<OutputT>::max())),
            static_cast<int32_t>(std::numeric_limits<OutputT>::min())));
      } else {
        output_data[i * depth + c] = std::numeric_limits<OutputT>::min();
      }
    }
  }
}

// Computes exp(input - max_input)
inline int16_t SoftMaxCalculateExp(const SoftmaxParams& params,
                                   const int16_t* input_data, const int depth,
                                   int16_t max_in_row, int i, int c) {
  int32_t input_diff = input_data[i * depth + c] - max_in_row;
  // scale the input_diff such that [-65535, 0] correspond to [-10.0, 0.0]
  // exp lut generated with range [-10, 0], as exp(-10) is negligible.
  int32_t scaled_diff = MultiplyByQuantizedMultiplier(
      input_diff, params.input_multiplier, params.input_left_shift);
  // recenter to [-32768, 32767]
  int32_t sym_scaled_diff = scaled_diff + 32767;
  int16_t sat_sym_scaled_diff =
      std::min(std::max(sym_scaled_diff, static_cast<int32_t>(-32768)),
               static_cast<int32_t>(32767));
  // apply the exp() LUT activation function
  return generic_int16_table_lookup(sat_sym_scaled_diff, params.exp_lut);
}
// Quantized softmax with int16_t input and int16_t output.
inline void SoftmaxInt16(const SoftmaxParams& params,
                         const RuntimeShape& input_shape,
                         const int16_t* input_data,
                         const RuntimeShape& output_shape,
                         int16_t* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    // Find the largest element
    int16_t max_in_row = std::numeric_limits<int16_t>::min();
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    // This loops computes the exp values and their sum. We will need the exp
    // values later on in the function so we cache them in the output_data
    // buffer. This is an optimization done to avoid calculating the exp values
    // twice making use of the output_data buffer as scratch memory.
    int32_t sum_of_exps = 0;  // Q16.15 fixed point format.
    int16_t* exp_results_Q015 = output_data + i * depth;
    for (int c = 0; c < depth; ++c) {
      exp_results_Q015[c] =
          SoftMaxCalculateExp(params, input_data, depth, max_in_row, i, c);
      sum_of_exps += exp_results_Q015[c];
    }

    // Compute the reciprocal 1/sum_of_exps
    uint8_t headroom_plus_one =
        CountLeadingZeros(static_cast<uint32_t>(sum_of_exps));
    int32_t shifted_sum =
        ((static_cast<int64_t>(sum_of_exps) << (headroom_plus_one - 1)) +
         (1 << 13)) >>
        14;
    // since the LUT computes 1/(1 + x) we need to first compute x = (sum - 1).
    // also, the LUT expects a symmetrical input, so we must also recenter x
    // from [0, 65535] to [-32768, 32767].
    int32_t sym_shifted_sum = shifted_sum + (-((1 << 15) + (1 << 16)));
    int16_t sat_sym_shifted_sum = static_cast<int16_t>(
        std::min(std::max(sym_shifted_sum, static_cast<int32_t>(-32768)),
                 static_cast<int32_t>(32767)));
    // apply 1/(1 + x) LUT activation function
    int16_t reciprocal_scale_Q015 = generic_int16_table_lookup(
        sat_sym_shifted_sum, params.one_over_one_plus_x_lut);

    // Rescale the exp_result with reciprocal
    // range of output is [0, 32767] correspond to [0.0, 1.0]
    for (int c = 0; c < depth; ++c) {
      uint8_t right_shift = 31 - headroom_plus_one;
      int64_t round = 1 << (right_shift - 1);
      int32_t result = (static_cast<int64_t>(exp_results_Q015[c]) *
                            static_cast<int64_t>(reciprocal_scale_Q015) +
                        round) >>
                       right_shift;
      output_data[i * depth + c] = static_cast<int16_t>(
          std::min(std::max(result, static_cast<int32_t>(0)),
                   static_cast<int32_t>(32767)));
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SOFTMAX_H_
