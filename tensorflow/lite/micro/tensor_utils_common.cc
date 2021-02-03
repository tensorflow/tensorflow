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

#include "tensorflow/lite/kernels/internal/tensor_utils_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"

namespace tflite {

//
// The following is copied from TfLite portable_tensor_utils.cc
//
// The declarations are located in header file:
//   tensorflow/lite/kernels/internal/tensor_utils_common.h
//
namespace tensor_utils {

// Quantizes a buffer of floating point values using a symmetric quantization
// (i.e. linear quantization without an offset) to 8-bit signed integers.
// It also outputs the range (min, max) of the floating point buffer, and the
// scaling factor used to quantize the values.
void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float* min_value,
                             float* max_value, float* scaling_factor) {
  auto minmax = std::minmax_element(values, values + size);
  *min_value = *minmax.first;
  *max_value = *minmax.second;

  SymmetricQuantizeFloats(values, size, quantized_values, *min_value,
                          *max_value, scaling_factor);
}

// Quantizes a buffer of floating point values using a symmetric quantization
// (i.e. linear quantization without an offset) to 8-bit signed integers.
// It uses the range (min, max) provided to the function to calculate the
// appropriate scaling factor to quantize the values.
void SymmetricQuantizeFloats(const float* values, const int size,
                             int8_t* quantized_values, float min_value,
                             float max_value, float* scaling_factor) {
  const int32_t kScale = 127;
  const float range = std::max(std::abs(min_value), std::abs(max_value));
  if (range == 0) {
    std::fill_n(quantized_values, size, 0);
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value =
        static_cast<int32_t>(TfLiteRound(values[i] * scaling_factor_inv));
    // Clamp: just in case some odd numeric offset.
    quantized_values[i] = static_cast<int8_t>(
        std::min(kScale, std::max(-kScale, quantized_value)));
  }
}

void AsymmetricQuantizeFloats(const float* values, const int size,
                              int8_t* quantized_values, float* scaling_factor,
                              int32_t* offset) {
  const int32_t kMinScale = -128;
  const int32_t kMaxScale = 127;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;
  const auto minmax = std::minmax_element(values, values + size);
  const double rmin = std::fmin(0, *minmax.first);
  const double rmax = std::fmax(0, *minmax.second);
  if (rmin == rmax) {
    std::fill_n(quantized_values, size, 0);
    *scaling_factor = 1;
    *offset = 0;
    return;
  } else {
    double scale = (rmax - rmin) / (qmax_double - qmin_double);
    const double zero_point_from_min = qmin_double - rmin / scale;
    const double zero_point_from_max = qmax_double - rmax / scale;
    const double zero_point_from_min_error =
        std::abs(qmin_double) + std::abs(rmin / scale);
    const double zero_point_from_max_error =
        std::abs(qmax_double) + std::abs(rmax / scale);
    const double zero_point_double =
        zero_point_from_min_error < zero_point_from_max_error
            ? zero_point_from_min
            : zero_point_from_max;
    int8_t nudged_zero_point = 0;
    if (zero_point_double <= qmin_double) {
      nudged_zero_point = kMinScale;
    } else if (zero_point_double >= qmax_double) {
      nudged_zero_point = kMaxScale;
    } else {
      nudged_zero_point = static_cast<int8_t>(round(zero_point_double));
    }
    *scaling_factor = scale;
    *offset = nudged_zero_point;
  }
  const float scaling_factor_inv = 1.0f / *scaling_factor;
  for (int i = 0; i < size; ++i) {
    const int32_t quantized_value = static_cast<int32_t>(
        TfLiteRound(*offset + values[i] * scaling_factor_inv));
    quantized_values[i] =
        std::min(kMaxScale, std::max(kMinScale, quantized_value));
  }
}

// Reduce-sum on a vector:
// input_vector: pointer to input vector.
// output_vector: pointer to vector.
// output_size: output vector size.
// reduction_size: number of consecutive elements from input vector which are
// added to get one element of output.
void ReductionSumVector(const int8_t* input_vector, int32_t* output_vector,
                        int output_size, int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    int32_t result = 0;
    for (int r = 0; r < reduction_size; r++) {
      result += input_vector[r];
    }
    output_vector[o] = result;
    input_vector += reduction_size;
  }
}

}  // namespace tensor_utils

}  // namespace tflite
