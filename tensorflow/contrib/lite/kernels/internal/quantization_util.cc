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
#include <algorithm>
#include <cmath>
#include <limits>

#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/round.h"

namespace tflite {

void QuantizeMultiplierSmallerThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* right_shift) {
  TFLITE_CHECK(double_multiplier >= 0.);
  TFLITE_CHECK(double_multiplier < 1.);
  if (double_multiplier == 0.) {
    *quantized_multiplier = 0;
    *right_shift = 0;
    return;
  }
  TFLITE_CHECK(double_multiplier > 0.);
  const double q = std::frexp(double_multiplier, right_shift);
  *right_shift *= -1;

  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1ll << 31)));
  TFLITE_CHECK(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    --*right_shift;
  }
  TFLITE_CHECK_GE(*right_shift, 0);
  TFLITE_CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t* quantized_multiplier,
                                      int* left_shift) {
  TFLITE_CHECK(double_multiplier > 1.);
  const double q = std::frexp(double_multiplier, left_shift);
  auto q_fixed = static_cast<int64_t>(TfLiteRound(q * (1ll << 31)));
  TFLITE_CHECK(q_fixed <= (1ll << 31));
  if (q_fixed == (1ll << 31)) {
    q_fixed /= 2;
    ++*left_shift;
  }
  TFLITE_CHECK_GE(*left_shift, 0);
  TFLITE_CHECK_LE(q_fixed, std::numeric_limits<int32_t>::max());
  *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void PreprocessSoftmaxScaling(double beta, double input_scale,
                              int input_integer_bits,
                              int32_t* quantized_multiplier, int* left_shift) {
  // If the overall multiplier (input and beta) is large, then exp() of an
  // input difference of 1 scaled by this will be large.  In other words, we
  // can cap the multiplier and know that, when it is used, the output will be
  // (round to) zero wherever the input is not at the maximum value.

  // If the overall scale is less than one, and input_integer_bits=0, then the
  // result is double equivalent of Q0.31 (actually with more precision). Thus
  // this generates a Q(input_integer_bits).(31-input_integer_bits)
  // representation.
  const double input_beta_real_multiplier = std::min(
      beta * input_scale * (1 << (31 - input_integer_bits)), (1ll << 31) - 1.0);

  QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier,
                                   quantized_multiplier, left_shift);
}

int CalculateInputRadius(int input_integer_bits, int input_left_shift) {
  const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                    (1ll << (31 - input_integer_bits)) /
                                    (1ll << input_left_shift);
  // Tighten bound using floor.  Suppose that we could use the exact value.
  // After scaling the difference, the result would be at the maximum.  Thus we
  // must ensure that our value has lower magnitude.
  return static_cast<int>(std::floor(max_input_rescaled));
}

}  // namespace tflite
