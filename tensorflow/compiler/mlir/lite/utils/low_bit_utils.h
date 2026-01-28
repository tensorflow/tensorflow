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
// Assumes that `src_tensor` is a buffer where each element is a low bit value
// (e.g. 2 or 4-bit) stored in 8-bit.
// Returns a new buffer that is packed densely.
// The packing format is low-bits-first.
std::vector<uint8_t> PackLowBitValuesDensely(std::vector<uint8_t> src_buffer,
                                             int bit_width);

// Assumes `src_buffer` contains densely packed low bit elements.
// Returns a vector where each int8 element contains a sign-extended value.
std::vector<char> UnpackDenseLowBitIntoInt8(
    const std::vector<uint8_t>& src_buffer, int64_t num_elements,
    int bit_width);

// Assumes `src_buffer` contains densely packed low bit elements.
// Returns a vector where each uint8 element contains an unpacked value.
std::vector<char> UnpackDenseLowBitIntoUint8(
    const std::vector<uint8_t>& src_buffer, int64_t num_elements,
    int bit_width);
}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_UTILS_LOW_BIT_UTILS_H_
