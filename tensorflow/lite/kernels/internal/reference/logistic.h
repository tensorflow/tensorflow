/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOGISTIC_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOGISTIC_H_

#include <cmath>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace reference_ops {

template <typename T>
inline void Logistic(const RuntimeShape& input_shape, const T* input_data,
                     const RuntimeShape& output_shape, T* output_data) {
  const float cutoff_upper = 16.619047164916992188f;
  const float cutoff_lower = -9.f;

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // Rational for using approximation in reference kernel.
  // 0. This approximation gives enough precision for float.
  // 1. This works around an issue on an embedded chipset where exp() does not
  // return correctly as expected - exp(x) should return inf when overflown
  // not 1.701417   IEEE 754 defines representation for inf.
  // 2. This will speed up calculation and is matching the behavior in the
  // optimized kernels. (check the definition of scalar_logistic_op<float>)

  for (int i = 0; i < flat_size; i++) {
    T val = input_data[i];
    float result;
    if (val > cutoff_upper) {
      result = 1.0f;
    } else if (val < cutoff_lower) {
      result = std::exp(val);
    } else {
      result = 1.f / (1.f + std::exp(-val));
    }
    output_data[i] = static_cast<T>(result);
  }
}

// Convenience version that allows, for example, generated-code calls to be
// uniform between data types.
inline void Logistic(const LogisticParams&, const RuntimeShape& input_shape,
                     const float* input_data, const RuntimeShape& output_shape,
                     float* output_data) {
  // Drop params: not needed.
  Logistic(input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const LogisticParams& params,
                     const RuntimeShape& input_shape, const int16_t* input_data,
                     const RuntimeShape& output_shape, int16_t* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

    const F3 input = F3::FromRaw(input_data[i]);
    F0 output = gemmlowp::logistic(input);
    output_data[i] = output.raw();
  }
}

// Quantized int8_t logistic activation.  Cheats by dequantizing and
// requantizing around the floating point logistic method.  This implementation
// is slow on platforms without a floating point unit.

// TODO(b/141211002): Delete this int8_t implementation once we can reuse the
// approach used in TFLite for int8_t Logistic.
inline void Logistic(const RuntimeShape& input_shape, const int8_t* input_data,
                     float input_scale, int input_zero_point,
                     const RuntimeShape& output_shape, int8_t* output_data,
                     float output_scale, int output_zero_point) {
  const float cutoff_upper = 16.619047164916992188f;
  const float cutoff_lower = -9.f;

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // Rational for using approximation in reference kernel.
  // 0. This approximation gives enough precision for float.
  // 1. This works around an issue on an embedded chipset where exp() does not
  // return correctly as expected - exp(x) should return inf when overflown
  // not 1.701417   IEEE 754 defines representation for inf.
  // 2. This will speed up calculation and is matching the behavior in the
  // optimized kernels. (check the definition of scalar_logistic_op<float>)

  for (int i = 0; i < flat_size; i++) {
    // Dequantize.
    float val =
        static_cast<float>((input_data[i] - input_zero_point) * input_scale);
    float result;
    if (val > cutoff_upper) {
      result = 1.0f;
    } else if (val < cutoff_lower) {
      result = std::exp(val);
    } else {
      result = 1.f / (1.f + std::exp(-val));
    }
    // Requantize
    int8_t output =
        static_cast<int8_t>(result / output_scale + output_zero_point);
    output_data[i] = output;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOGISTIC_H_
