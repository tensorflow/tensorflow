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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LOW_BIT_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LOW_BIT_UTILS_H_

#include <cstdint>
#include <vector>

namespace tflite {
// Assumes that `src_tensor` is a buffer where each element is a 4-bit value
// stored in 8-bit.
// Returns a new buffer that is packed densely with 2 4-bit values in a byte.
// The packing format is low-bits-first, i.e. the lower nibble of a byte is
// filled first, followed by the upper nibble.
std::vector<uint8_t> PackInt4ValuesDensely(std::vector<uint8_t> src_buffer);

// Assumes `src_buffer` contains 2 4-bit elements packed in 8-bit.
// Returns a vector where each int8 element contains a int4 sign-extended value.
std::vector<char> UnpackDenseInt4IntoInt8(
    const std::vector<uint8_t>& src_buffer, int64_t num_elements);
}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LOW_BIT_UTILS_H_
