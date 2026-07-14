/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/low_bit_utils.h"

#include <cassert>
#include <cstdint>
#include <vector>

namespace tflite {

std::vector<char> UnpackDenseLowBitIntoInt8(
    const std::vector<uint8_t>& src_buffer, int64_t num_elements,
    int bit_width) {
  std::vector<char> unpacked_buffer;
  unpacked_buffer.reserve(num_elements);
  const int elements_per_byte = 8 / bit_width;
  const int sign_bit_shift = 8 - bit_width;
  const uint8_t mask = (1 << bit_width) - 1;

  for (uint8_t value : src_buffer) {
    for (int i = 0; i < elements_per_byte; ++i) {
      if (unpacked_buffer.size() == num_elements) break;
      int bit_offset = i * bit_width;
      uint8_t extracted_value = (value >> bit_offset) & mask;
      // Sign extend
      unpacked_buffer.push_back(
          static_cast<int8_t>(extracted_value << sign_bit_shift) >>
          sign_bit_shift);
    }
  }

  return unpacked_buffer;
}

std::vector<char> UnpackDenseLowBitIntoUint8(
    const std::vector<uint8_t>& src_buffer, int64_t num_elements,
    int bit_width) {
  std::vector<char> unpacked_buffer;
  unpacked_buffer.reserve(num_elements);
  const int elements_per_byte = 8 / bit_width;
  const uint8_t mask = (1 << bit_width) - 1;

  for (uint8_t value : src_buffer) {
    for (int i = 0; i < elements_per_byte; ++i) {
      if (unpacked_buffer.size() == num_elements) break;
      int bit_offset = i * bit_width;
      unpacked_buffer.push_back((value >> bit_offset) & mask);
    }
  }

  return unpacked_buffer;
}

}  // namespace tflite
