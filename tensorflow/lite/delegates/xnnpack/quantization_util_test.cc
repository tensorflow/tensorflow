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

#include "tensorflow/lite/delegates/xnnpack/quantization_util.h"

#include <stdint.h>

#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/types.h"

using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace xnnpack {
namespace {

template <typename T>
inline double ScaleFromMinMax(const float min, const float max) {
  return (max - min) / ((std::numeric_limits<T>::max() * 1.0) -
                        std::numeric_limits<T>::min());
}

template <typename T>
inline int32_t ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int32_t>(std::numeric_limits<T>::min()) +
         static_cast<int32_t>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

TEST(Dequantize, Int8) {
  std::vector<int8_t> quantized_data = {-3, -2, -1, 1, 2, 3};
  std::vector<float> dequantized_data(quantized_data.size());

  RuntimeShape tensor_shape(1, quantized_data.size());

  const float min = -12.8f;
  const float max = 12.7f;

  const double scale = ScaleFromMinMax<int8_t>(min, max);
  const int32_t zero_point = ZeroPointFromMinMax<int8_t>(min, max);

  DequantizeInt8(quantized_data.data(), dequantized_data.data(), tensor_shape,
                 zero_point, scale);
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3}));
}

TEST(Dequantize, Float16) {
  std::vector<uint16_t> quantized_data = {
      UINT16_C(0x3000),  // 0.125
      UINT16_C(0x3400),  // 0.25
      UINT16_C(0x3800),  // 0.5
      UINT16_C(0x3C00),  // 1
      UINT16_C(0x4000),  // 2
      UINT16_C(0x4400)   // 4
  };
  std::vector<float> dequantized_data(quantized_data.size());

  DequantizeFloat16(quantized_data.data(), dequantized_data.data(),
                    quantized_data.size());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {0.125, 0.25, 0.5, 1., 2., 4.}));
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
