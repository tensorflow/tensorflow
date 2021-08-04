/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOG_SOFTMAX_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOG_SOFTMAX_H_

#include <algorithm>
#include <cstddef>
#include <limits>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

inline void LogSoftmax(const SoftmaxParams& params,
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
    // log(exp(x[i])/sum(exp(x[i]))) == log(exp(x[i]+C)/sum(exp(x[i]+C)))
    float max = std::numeric_limits<float>::lowest();
    for (int c = 0; c < depth; ++c) {
      max = std::max(max, input_data[i * depth + c]);
    }

    // Compute sum.
    float sum = 0.f;
    for (int c = 0; c < depth; ++c) {
      sum += std::exp(input_data[i * depth + c] - max);
    }

    // Compute result.
    const float log_sum = std::log(sum);
    for (int c = 0; c < depth; ++c) {
      output_data[i * depth + c] = input_data[i * depth + c] - max - log_sum;
    }
  }
}

inline void LogSoftmax(const SoftmaxParams& params,
                       const RuntimeShape& input_shape,
                       const uint8_t* input_data,
                       const RuntimeShape& output_shape, uint8_t* output_data) {
  const int32_t input_multiplier = params.input_multiplier;
  const int32_t input_left_shift = params.input_left_shift;
  const int32_t reverse_scaling_divisor = params.reverse_scaling_divisor;
  const int32_t reverse_scaling_right_shift =
      params.reverse_scaling_right_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large
  // as -32 before multiplying by input_beta_multiplier, and therefore as
  // large as -16 afterwards.  Note that exp(-8) is definitely not
  // insignificant to accumulation, but exp(-16) definitely is.
  static constexpr int kScaledDiffIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
  using FixedPointAccum =
      gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    uint8_t max_in_row = 0;
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
                input_diff, input_multiplier, input_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    const int32_t fixed_log_sum_of_exps =
        log_x_for_x_greater_than_or_equal_to_1<kScaledDiffIntegerBits>(
            sum_of_exps)
            .raw();

    // rescaled_diff_min is smallest representable in
    // Q(kScaledDiffIntegerBits).(31-kScaledDiffIntegerBits) plus the
    // log-sub-exps that will be subtracted in the loop.
    //
    // The thresholds diff_min, etc are negative.
    const int rescaled_diff_min =
        fixed_log_sum_of_exps + std::numeric_limits<int32_t>::lowest();
    const int adjusted_diff_min =
        std::max(static_cast<int32_t>(
                     diff_min - 1),  // Note use of > below instead of >= above.
                 MultiplyByQuantizedMultiplierSmallerThanOneExp(
                     rescaled_diff_min, reverse_scaling_divisor,
                     -reverse_scaling_right_shift));

    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff > adjusted_diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
        int32_t unsat_output =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_rescaled - fixed_log_sum_of_exps),
                31 - kScaledDiffIntegerBits - kOutputIntegerBits) +
            255;

        output_data[i * depth + c] = static_cast<uint8_t>(
            std::max(std::min(unsat_output, static_cast<int32_t>(255)),
                     static_cast<int32_t>(0)));
      } else {
        // Set output to smallest value.
        output_data[i * depth + c] = 0;
      }
    }
  }
}

template <typename T>
inline void LogSoftmaxQuantized(const SoftmaxParams& params,
                                const size_t outer_size, const size_t depth,
                                const RuntimeShape& input_shape,
                                const T* input_data,
                                const RuntimeShape& output_shape,
                                T* output_data) {
  const int32_t input_multiplier = params.input_multiplier;
  const int32_t input_left_shift = params.input_left_shift;
  const int32_t reverse_scaling_divisor = params.reverse_scaling_divisor;
  const int32_t reverse_scaling_right_shift =
      params.reverse_scaling_right_shift;
  const int diff_min = params.diff_min;

  static constexpr T kMinT8 = std::numeric_limits<T>::min();
  static constexpr T kMaxT8 = std::numeric_limits<T>::max();
  static constexpr int32_t kMinInt32 = std::numeric_limits<int32_t>::min();

  // All IntegerBits must agree with Prepare function.
  // Input is chosen as Q5.26 so exp(-1 * 2^5 * 2^-1) = exp(-16) is negligible.
  static constexpr int kInputIntegerBits = 5;
  static constexpr int kAccumulationIntegerBits = 12;
  static constexpr int kOutputIntegerBits = 4;
  using F5 = gemmlowp::FixedPoint<int32_t, kInputIntegerBits>;
  using F12 = gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;

  for (size_t outer_index = 0; outer_index < outer_size; ++outer_index) {
    T max_in_row = kMinT8;
    for (size_t inner_index = 0; inner_index < depth; ++inner_index) {
      max_in_row =
          std::max(max_in_row, input_data[outer_index * depth + inner_index]);
    }

    // Accumulator "sum_of_exps_in_q12" is safe from overflowing in 2^12 steps.
    F12 sum_of_exps_in_q12 = F12::FromRaw(0);
    for (size_t inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[outer_index * depth + inner_index]) -
          max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_in_q5 = MultiplyByQuantizedMultiplier(
            input_diff, input_multiplier, input_left_shift);
        sum_of_exps_in_q12 =
            sum_of_exps_in_q12 +
            gemmlowp::Rescale<kAccumulationIntegerBits>(
                exp_on_negative_values(F5::FromRaw(input_diff_in_q5)));
      }
    }

    const int32_t log_sum_of_exps_in_q5 =
        log_x_for_x_greater_than_or_equal_to_1<kInputIntegerBits>(
            sum_of_exps_in_q12)
            .raw();

    // Potentially reduced the valid range. shifted_log_sum_of_exps_in_q5 is
    // smallest representable in Q5.26 plus the log_sum_of_exps.
    const int32_t shifted_log_sum_of_exps_in_q5 =
        log_sum_of_exps_in_q5 + kMinInt32;
    const int32_t adjusted_diff_min =
        std::max(static_cast<int32_t>(diff_min - 1),
                 MultiplyByQuantizedMultiplier(shifted_log_sum_of_exps_in_q5,
                                               reverse_scaling_divisor,
                                               -reverse_scaling_right_shift));

    for (size_t inner_index = 0; inner_index < depth; ++inner_index) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[outer_index * depth + inner_index]) -
          max_in_row;
      // Note use of > below instead of >= above.
      if (input_diff > adjusted_diff_min) {
        const int32_t input_diff_in_q5 = MultiplyByQuantizedMultiplier(
            input_diff, input_multiplier, input_left_shift);

        // Rescale and downcast.
        int32_t output_in_q27 =
            gemmlowp::RoundingDivideByPOT(
                (input_diff_in_q5 - log_sum_of_exps_in_q5),
                31 - kInputIntegerBits - kOutputIntegerBits) +
            kMaxT8;

        output_in_q27 =
            std::max(std::min(output_in_q27, static_cast<int32_t>(kMaxT8)),
                     static_cast<int32_t>(kMinT8));
        output_data[outer_index * depth + inner_index] =
            static_cast<T>(output_in_q27);
      } else {
        output_data[outer_index * depth + inner_index] = kMinT8;
      }
    }
  }
}

inline void LogSoftmax(const SoftmaxParams& params, const size_t outer_size,
                       const size_t depth, const RuntimeShape& input_shape,
                       const int8_t* input_data,
                       const RuntimeShape& output_shape, int8_t* output_data) {
  LogSoftmaxQuantized(params, outer_size, depth, input_shape, input_data,
                      output_shape, output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOG_SOFTMAX_H_
