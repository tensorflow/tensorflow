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

#include "tensorflow/lite/delegates/gpu/metal/kernels/util.h"

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/types.h"

namespace tflite {
namespace gpu {
namespace metal {

/// Converts float to destination type (if needed) and stores as bytes array.
std::vector<uint8_t> GetByteBufferConverted(
    const std::vector<float>& input_vector, DataType data_type) {
  if (data_type == DataType::FLOAT32) {
    return GetByteBuffer(input_vector);
  } else {
    std::vector<uint8_t> result;
    result.reserve(input_vector.size() * sizeof(half));
    for (const float value : input_vector) {
      const half converted = half(value);
      const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&converted);
      result.insert(result.end(), bytes, bytes + sizeof(half));
    }
    return result;
  }
}

/// Resizes, Converts float to destination type (if needed) and stores as bytes
/// array.
std::vector<uint8_t> GetByteBufferConvertedResized(
    const std::vector<float>& input_vector, DataType data_type,
    size_t elements_count) {
  auto result = GetByteBufferConverted(input_vector, data_type);
  const size_t type_size =
      data_type == DataType::FLOAT32 ? sizeof(float) : sizeof(half);
  result.resize(type_size * elements_count);
  return result;
}

}  // namespace metal
}  // namespace gpu
}  // namespace tflite
