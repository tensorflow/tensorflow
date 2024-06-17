/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/utils/string_utils.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mlir::TFL {

bool SimpleDynamicBuffer::AddString(const char* str, size_t len) {
  // If `data_.size() + len` is greater than `SIZE_MAX` then the left hand side
  // will overflow to something less than max_length_. After checking `len <=
  // max_length_` we can use this subtraction to check for overflow.
  if (len > max_length_ || data_.size() >= max_length_ - len) return false;
  data_.resize(data_.size() + len);
  memcpy(data_.data() + offset_.back(), str, len);
  offset_.push_back(offset_.back() + len);
  return true;
}

int SimpleDynamicBuffer::WriteToBuffer(char** buffer) {
  // Allocate sufficient memory to tensor buffer.
  int32_t num_strings = offset_.size() - 1;
  // Total bytes include:
  //   * size of content (data_.size)
  //   * offset of each tensor (sizeof(int32_t) * num_strings)
  //   * length of whole buffer (int32_t)
  //   * num of strings (int32_t).
  int32_t bytes = data_.size()                            // size of content
                  + sizeof(int32_t) * (num_strings + 2);  // size of header
  // Caller will take ownership of buffer.
  *buffer = reinterpret_cast<char*>(malloc(bytes));

  if (*buffer == nullptr) {
    return -1;
  }

  // Set num of string
  //
  // NOTE: The string buffer is accessed here as if it's native endian (instead
  // of small endian, as documented in the header). This will protentially break
  // when TFLite is ported to big endian platforms.
  // TODO(b/165919229): This code will need changing if/when we port to a
  // big-endian platform.
  memcpy(*buffer, &num_strings, sizeof(int32_t));

  // Set offset of strings.
  int32_t start = sizeof(int32_t) * (num_strings + 2);
  for (size_t i = 0; i < offset_.size(); i++) {
    // TODO(b/165919229): This code will need changing if/when we port to a
    // big-endian platform.
    int32_t offset = start + offset_[i];
    memcpy(*buffer + sizeof(int32_t) * (i + 1), &offset, sizeof(int32_t));
  }

  // Copy data of strings.
  memcpy(*buffer + start, data_.data(), data_.size());
  return bytes;
}

}  // namespace mlir::TFL
