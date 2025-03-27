// Copyright (c) Qualcomm Innovation Center, Inc. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/core/utils/miscs.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"

namespace qnn {
void ConvertDataFromInt16toUInt16(absl::Span<const std::int16_t> src,
                                  std::vector<std::uint16_t>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (const auto& data : src) {
    dst.emplace_back(data + kUint16ZeroPoint);
  }
}

void ConvertDataFromUInt16toInt16(absl::Span<const std::uint16_t> src,
                                  std::vector<std::int16_t>& dst) {
  dst.clear();
  dst.reserve(src.size());
  for (const auto& data : src) {
    dst.emplace_back(data - kUint16ZeroPoint);
  }
}

void ConvertDataFromInt4ToInt8(const void* src, std::vector<std::int8_t>& dst,
                               size_t num_bytes) {
  dst.clear();
  const uint8_t* byte_data = reinterpret_cast<const uint8_t*>(src);
  for (size_t i = 0; i < num_bytes; i++) {
    uint8_t byte = byte_data[i];
    // Extract lower and upper 4-bit values
    int8_t lower = byte & 0x0F;
    int8_t upper = (byte >> 4) & 0x0F;
    // Sign extend if needed
    if (lower > 7) lower -= 16;
    if (upper > 7) upper -= 16;
    // Store in output array
    dst.emplace_back(lower);
    dst.emplace_back(upper);
  }
}

}  // namespace qnn
