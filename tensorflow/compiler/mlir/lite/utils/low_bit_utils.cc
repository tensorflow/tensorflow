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

std::vector<uint8_t> PackInt4ValuesDensely(std::vector<uint8_t> src_buffer) {
  std::vector<uint8_t> packed_buffer((src_buffer.size() + 1) / 2);

  for (int i = 0; i < src_buffer.size(); ++i) {
    if (i % 2 == 0) {
      packed_buffer.at(i / 2) = src_buffer[i];
    } else {
      packed_buffer.at(i / 2) |= src_buffer[i] << 4;
    }
  }

  return packed_buffer;
}

std::vector<char> UnpackDenseInt4IntoInt8(
    const std::vector<uint8_t>& src_buffer, int64_t num_elements) {
  std::vector<char> unpacked_buffer;
  unpacked_buffer.reserve(num_elements);

  for (uint8_t value : src_buffer) {
    // Cast to signed before right-shifting to ensure correct sign extension
    unpacked_buffer.push_back(static_cast<int8_t>(value << 4) >> 4);
    unpacked_buffer.push_back(static_cast<int8_t>(value) >> 4);
  }

  // The last element might be a padded zero, so check and pop if needed
  if (unpacked_buffer.size() > num_elements) {
    assert(unpacked_buffer.size() == num_elements + 1);
    unpacked_buffer.pop_back();
  }

  return unpacked_buffer;
}

}  // namespace tflite
