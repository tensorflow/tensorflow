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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_UTIL_H_

#include <vector>

#include "tensorflow/lite/delegates/gpu/common/data_type.h"

namespace tflite {
namespace gpu {
namespace metal {

/// Helper function to convert buffer's content into stream of bytes
template <typename T>
std::vector<uint8_t> GetByteBuffer(const std::vector<T>& input_vector) {
  std::vector<uint8_t> result;
  result.insert(result.begin(),
                reinterpret_cast<const uint8_t*>(input_vector.data()),
                reinterpret_cast<const uint8_t*>(input_vector.data()) +
                    input_vector.size() * sizeof(*input_vector.data()));
  return result;
}

/// Converts float to destination type (if needed) and stores as bytes array.
/// supports DataType::FLOAT32 and DataType::FLOAT16
std::vector<uint8_t> GetByteBufferConverted(
    const std::vector<float>& input_vector, DataType data_type);

/// Resizes, Converts float to destination type (if needed) and stores as bytes
/// array.
/// supports DataType::FLOAT32 and DataType::FLOAT16
std::vector<uint8_t> GetByteBufferConvertedResized(
    const std::vector<float>& input_vector, DataType data_type,
    size_t elements_count);

}  // namespace metal
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_METAL_KERNELS_UTIL_H_
