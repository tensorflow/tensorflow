/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TANH_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TANH_H_

#include <cmath>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace reference_ops {

inline void Tanh(const RuntimeShape& input_shape, const float* input_data,
                 const RuntimeShape& output_shape, float* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    float val = input_data[i];
    float result = std::tanh(val);
    output_data[i] = result;
  }
}

// Convenience version that allows, for example, generated-code calls to be
// uniform between data types.
inline void Tanh(const TanhParams&, const RuntimeShape& input_shape,
                 const float* input_data, const RuntimeShape& output_shape,
                 float* output_data) {
  // Drop params: not needed.
  Tanh(input_shape, input_data, output_shape, output_data);
}

inline void Tanh(const TanhParams& params, const RuntimeShape& input_shape,
                 const int16* input_data, const RuntimeShape& output_shape,
                 int16* output_data) {
  const int input_left_shift = params.input_left_shift;
  // Support for shifts is limited until we have a parameterized version of
  // SaturatingRoundingMultiplyByPOT().
  TFLITE_DCHECK_GE(input_left_shift, 0);
  TFLITE_DCHECK_LE(input_left_shift, 1);

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // F0 uses 0 integer bits, range [-1, 1].
  // This is the return type of math functions such as tanh, logistic,
  // whose range is in [-1, 1].
  using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
  // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
  using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

  if (input_left_shift == 0) {
    for (int i = 0; i < flat_size; i++) {
      F3 input = F3::FromRaw(input_data[i]);
      F0 output = gemmlowp::tanh(input);
      output_data[i] = output.raw();
    }
  } else {
    for (int i = 0; i < flat_size; i++) {
      F3 input = F3::FromRaw(
          gemmlowp::SaturatingRoundingMultiplyByPOT<1>(input_data[i]));
      F0 output = gemmlowp::tanh(input);
      output_data[i] = output.raw();
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_TANH_H_
