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

#include "tensorflow/lite/delegates/xnnpack/test_util.h"

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/cppmath.h"

namespace tflite {
namespace xnnpack {

int8_t QuantizeInt8(float value, int32_t zero_point, float scale) {
  static constexpr int32_t min_val = std::numeric_limits<int8_t>::min();
  static constexpr int32_t max_val = std::numeric_limits<int8_t>::max();

  int32_t unclamped =
      static_cast<int32_t>(TfLiteRound(value / scale)) + zero_point;
  int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
  return static_cast<int8_t>(clamped);
}

float GetInt8QuantizationScale(const std::vector<float>& data) {
  static constexpr int8_t qmin_val = std::numeric_limits<int8_t>::min();
  static constexpr int8_t qmax_val = std::numeric_limits<int8_t>::max();
  static constexpr float qmin_float = qmin_val;
  static constexpr float qmax_float = qmax_val;

  return (*std::max_element(data.begin(), data.end()) -
          *std::min_element(data.begin(), data.end())) /
         (qmax_float - qmin_float);
}

}  // namespace xnnpack
}  // namespace tflite
