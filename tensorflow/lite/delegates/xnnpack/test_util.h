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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_TEST_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_TEST_UTIL_H_

#include <cstdint>
#include <vector>

namespace tflite {
namespace xnnpack {

int8_t QuantizeInt8(float value, int32_t zero_point, float scale);

void QuantizeInt8PerChannel(const float* scale, const int64_t* zero_point,
                            int32_t quantized_dimension,
                            const float* input_data, int8_t* output_data,
                            const std::vector<int32_t>& shape);

float GetInt8QuantizationScale(const std::vector<float>& data);

std::vector<float> GetInt8QuantizationScalePerChannel(
    const float* data, int32_t quantized_dimension,
    const std::vector<int32_t>& shape);

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_TEST_UTIL_H_
