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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_STRING_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_STRING_UTILS_H_

#include <stdint.h>

#include <cstddef>
#include <limits>
#include <vector>

#include "absl/status/status.h"

namespace mlir::TFL {

constexpr uint64_t kDefaultMaxLength = std::numeric_limits<int>::max();

class MiniDynamicBuffer {
 public:
  explicit MiniDynamicBuffer(size_t max_length = kDefaultMaxLength)
      : offset_({0}), max_length_(max_length) {}

  // Add string to dynamic buffer by resizing the buffer and copying the data.
  absl::Status AddString(const char* str, size_t len);

  // Fill content into a buffer and returns the number of bytes stored.
  // The function allocates space for the buffer but does NOT take ownership.
  int WriteToBuffer(char** buffer);

 private:
  // Data buffer to store contents of strings, not including headers.
  std::vector<char> data_;
  // Offset of the starting index of each string in data buffer.
  std::vector<size_t> offset_;
  // Max length in number of characters that we permit the total
  // buffer containing the concatenation of all added strings to be.
  // For historical reasons this is limited to 32bit length. At this files
  // inception, sizes were represented using 32bit which forced an implicit cap
  // on the size of the buffer. When this was refactored to use size_t (which
  // could be 64bit) we enforce that the buffer remains at most 32bit length to
  // avoid a change in behavior.
  const size_t max_length_;
};
}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_STRING_UTILS_H_
