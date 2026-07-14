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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace tflite {
// Packs raw byte inputs (chemically containing 2-bit or 4-bit values) into a
// dense bit-packed representation using Little-Endian (low-bits-first) order.
// The input buffer must contain unpacked values (1 value per byte).
//
// Requirement: 8 must be evenly divisible by kBitWidth.
//
// Template Args:
//   kBitWidth: The logical bit width of the values (must be 2 or 4).
template <int kBitWidth>
absl::Status StreamPackLowBitValues8Bit(
    llvm::ArrayRef<char> values,
    absl::FunctionRef<absl::Status(absl::string_view)> apply_chunk) {
  static_assert(kBitWidth == 2 || kBitWidth == 4);
  // Use a chunk size that balances overhead with cache locality.
  constexpr size_t kChunkSize = 1024 * 1024;

  std::vector<char> packed_buffer;

  constexpr size_t kElementsPerByte = 8 / kBitWidth;
  constexpr size_t kPackedBufferSize =
      (kChunkSize + kElementsPerByte - 1) / kElementsPerByte;
  packed_buffer.resize(kPackedBufferSize);

  // Outer Loop: Iterate over large chunks of the input
  for (size_t offset = 0; offset < values.size(); offset += kChunkSize) {
    llvm::ArrayRef<char> chunk_buffer =
        values.slice(offset, std::min(kChunkSize, values.size() - offset));

    const size_t num_elements = chunk_buffer.size();
    constexpr uint8_t mask = (1 << kBitWidth) - 1;
    size_t packed_idx = 0;

    // Inner Loop: Iterate through the chunk to pack bits
    for (size_t local_idx = 0; local_idx < num_elements;
         local_idx += kElementsPerByte) {
      uint8_t out_byte = 0;
      for (int j = 0; j < kElementsPerByte; ++j) {
        if (local_idx + j < num_elements) {
          out_byte |= (chunk_buffer[local_idx + j] & mask) << (j * kBitWidth);
        }
      }
      packed_buffer[packed_idx++] = static_cast<char>(out_byte);
    }

    auto status =
        apply_chunk(absl::string_view(packed_buffer.data(), packed_idx));
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

// Packs elements from a generic range (e.g., mlir::APInt from
// DenseElementsAttr) into a dense representation using little-endian
// (low-bits-first) bit-packing.
//
// Requirement: 8 must be evenly divisible by kBitWidth.
//
// Template Args:
//   kBitWidth: The logical bit width of the values (must be 2 or 4).
//   Range: A range type providing `begin()` and `end()` returning `APInt`.
template <int kBitWidth, typename Range>
absl::Status StreamPackLowBitValues(
    Range&& values,
    absl::FunctionRef<absl::Status(absl::string_view)> apply_chunk) {
  static_assert(kBitWidth == 2 || kBitWidth == 4);
  constexpr size_t kChunkSize = 1024 * 1024;

  std::vector<uint8_t> chunk_buffer;
  std::vector<char> packed_buffer;

  chunk_buffer.reserve(kChunkSize);

  constexpr size_t kElementsPerByte = 8 / kBitWidth;
  constexpr size_t kPackedBufferSize =
      (kChunkSize + kElementsPerByte - 1) / kElementsPerByte;
  packed_buffer.resize(kPackedBufferSize);

  auto it = values.begin();
  auto end = values.end();

  while (it != end) {
    // Phase 1: Unroll complex iterator into a simple contiguous buffer
    chunk_buffer.clear();
    for (size_t i = 0; i < kChunkSize && it != end; ++it, ++i) {
      chunk_buffer.push_back(static_cast<uint8_t>(*((*it).getRawData())));
    }

    const size_t num_elements = chunk_buffer.size();
    constexpr uint8_t mask = (1 << kBitWidth) - 1;
    size_t packed_idx = 0;

    // Phase 2: Pack bits (Auto-vectorized by compiler)
    for (size_t i = 0; i < num_elements; i += kElementsPerByte) {
      uint8_t out_byte = 0;
      for (int j = 0; j < kElementsPerByte; ++j) {
        if (i + j < num_elements) {
          out_byte |= (chunk_buffer[i + j] & mask) << (j * kBitWidth);
        }
      }
      packed_buffer[packed_idx++] = static_cast<char>(out_byte);
    }

    auto status =
        apply_chunk(absl::string_view(packed_buffer.data(), packed_idx));
    if (!status.ok()) {
      return status;
    }
  }

  return absl::OkStatus();
}

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
