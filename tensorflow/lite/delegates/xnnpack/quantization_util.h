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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZATION_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZATION_UTIL_H_

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace xnnpack {

// Dequantizes INT8 value using given zero point and scale.
// packed_s8_data should contain raw tensor data corresponding to
// a given tensor_shape. unpacked_fp32_data should be preallocated
// to have the same size.
void DequantizeInt8(const int8_t* packed_s8_data, float* unpacked_fp32_data,
                    const RuntimeShape& tensor_shape, int32_t zero_point,
                    double scale);

// Dequantizes INT8 value using given zero point and scale.
// packed_fp16_data should have tensor_elements size and contain raw
// FP16 tensor data. unpacked_fp32_data should be preallocated to
// have the same size.
void DequantizeFloat16(const uint16_t* packed_fp16_data,
                       float* unpacked_fp32_data, size_t tensor_elements);

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_QUANTIZATION_UTIL_H_
