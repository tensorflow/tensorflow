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

#include "tensorflow/lite/delegates/gpu/metal/compute_task_descriptor.h"

#include <cstdint>
#include <vector>

#include <fp16.h>
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/common/util.h"
#include "tensorflow/lite/delegates/gpu/metal/runtime_options.h"

namespace tflite {
namespace gpu {
namespace metal {

/// Converts float to destination type (if needed) and stores as bytes array.
std::vector<uint8_t> GetByteBufferConverted(
    const std::vector<float>& input_vector,
    RuntimeOptions::Precision destination_type) {
  if (destination_type == metal::RuntimeOptions::Precision::FP32) {
    return GetByteBuffer(input_vector);
  } else {
    std::vector<uint8_t> result;
    result.reserve(input_vector.size() * sizeof(HalfBits));
    for (const float value : input_vector) {
      const HalfBits converted = fp16_ieee_from_fp32_value(value);
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&converted);
      result.insert(result.end(), bytes, bytes + sizeof(HalfBits));
    }
    return result;
  }
}

/// Resizes, Converts float to destination type (if needed) and stores as bytes
/// array.
std::vector<uint8_t> GetByteBufferConvertedResized(
    const std::vector<float>& input_vector,
    RuntimeOptions::Precision destination_type, size_t elements_count) {
  auto result = GetByteBufferConverted(input_vector, destination_type);
  const size_t type_size =
      destination_type == metal::RuntimeOptions::Precision::FP32
          ? sizeof(float)
          : sizeof(HalfBits);
  result.resize(type_size * elements_count);
  return result;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
