/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/ir/tfl_ops_in_place_transpose.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace mlir {
namespace TFL {
namespace {

// Helper to read packed sub-byte element at index `idx` (matching packing: low
// bits first).
uint8_t ReadSubByte(const uint8_t* data, size_t idx, int bit_width) {
  size_t elems_per_byte = 8 / bit_width;
  size_t byte_idx = idx / elems_per_byte;
  size_t bit_offset = (idx % elems_per_byte) * bit_width;
  uint8_t mask = (1 << bit_width) - 1;
  return (data[byte_idx] >> bit_offset) & mask;
}

// Helper to write packed sub-byte element at index `idx`.
void WriteSubByte(uint8_t* data, size_t idx, uint8_t val, int bit_width) {
  size_t elems_per_byte = 8 / bit_width;
  size_t byte_idx = idx / elems_per_byte;
  size_t bit_offset = (idx % elems_per_byte) * bit_width;
  uint8_t mask = (1 << bit_width) - 1;
  data[byte_idx] =
      (data[byte_idx] & ~(mask << bit_offset)) | ((val & mask) << bit_offset);
}

TEST(TflOpsInPlaceTransposeTest, InvalidPermutationsReturnFalse) {
  std::vector<uint8_t> buffer(16, 0);
  int64_t shape[] = {2, 2};

  // Rank mismatch
  int64_t perm_rank_mismatch[] = {1};
  EXPECT_FALSE(InPlaceTranspose(buffer.data(), shape, perm_rank_mismatch, 8));
  EXPECT_FALSE(TransposeBuffer(buffer.data(), buffer.data(), shape,
                               perm_rank_mismatch, 8));

  // Out of bounds
  int64_t perm_out_of_bounds[] = {0, 2};
  EXPECT_FALSE(InPlaceTranspose(buffer.data(), shape, perm_out_of_bounds, 8));

  // Duplicate index
  int64_t perm_duplicate[] = {1, 1};
  EXPECT_FALSE(InPlaceTranspose(buffer.data(), shape, perm_duplicate, 8));

  // Negative index
  int64_t perm_negative[] = {-1, 0};
  EXPECT_FALSE(InPlaceTranspose(buffer.data(), shape, perm_negative, 8));

  // Unsupported bit width
  int64_t valid_perm[] = {1, 0};
  EXPECT_FALSE(InPlaceTranspose(buffer.data(), shape, valid_perm, 3));
  EXPECT_FALSE(InPlaceTranspose(buffer.data(), shape, valid_perm, 5));
}

TEST(TflOpsInPlaceTransposeTest, SubByte2DTranspose4Bit) {
  // Shape: 4 rows x 6 cols, bit_width = 4
  // Total elements = 24, total bytes = 12
  const int64_t R = 4;
  const int64_t C = 6;
  const int bit_width = 4;
  const size_t total_elements = R * C;
  const size_t byte_count = (total_elements * bit_width + 7) / 8;

  std::vector<uint8_t> src(byte_count, 0);
  for (size_t i = 0; i < total_elements; ++i) {
    WriteSubByte(src.data(), i, static_cast<uint8_t>(i % 16), bit_width);
  }

  // 1. Out-of-place TransposeBuffer
  int64_t shape[] = {R, C};
  int64_t perm[] = {1, 0};
  std::vector<uint8_t> dst(byte_count, 0);
  EXPECT_TRUE(TransposeBuffer(src.data(), dst.data(), shape, perm, bit_width));

  // Verify dst[c * R + r] == src[r * C + c]
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      uint8_t expected = ReadSubByte(src.data(), r * C + c, bit_width);
      uint8_t actual = ReadSubByte(dst.data(), c * R + r, bit_width);
      EXPECT_EQ(actual, expected) << "Mismatch at r=" << r << ", c=" << c;
    }
  }

  // 2. In-place InPlaceTranspose
  std::vector<uint8_t> in_place_buf = src;
  EXPECT_TRUE(InPlaceTranspose(in_place_buf.data(), shape, perm, bit_width));
  EXPECT_EQ(in_place_buf, dst);
}

TEST(TflOpsInPlaceTransposeTest, SubByte2DTranspose2Bit) {
  // Shape: 4 rows x 8 cols, bit_width = 2
  const int64_t R = 4;
  const int64_t C = 8;
  const int bit_width = 2;
  const size_t total_elements = R * C;
  const size_t byte_count = (total_elements * bit_width + 7) / 8;

  std::vector<uint8_t> src(byte_count, 0);
  for (size_t i = 0; i < total_elements; ++i) {
    WriteSubByte(src.data(), i, static_cast<uint8_t>(i % 4), bit_width);
  }

  int64_t shape[] = {R, C};
  int64_t perm[] = {1, 0};
  std::vector<uint8_t> dst(byte_count, 0);
  EXPECT_TRUE(TransposeBuffer(src.data(), dst.data(), shape, perm, bit_width));

  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      uint8_t expected = ReadSubByte(src.data(), r * C + c, bit_width);
      uint8_t actual = ReadSubByte(dst.data(), c * R + r, bit_width);
      EXPECT_EQ(actual, expected) << "Mismatch at r=" << r << ", c=" << c;
    }
  }

  std::vector<uint8_t> in_place_buf = src;
  EXPECT_TRUE(InPlaceTranspose(in_place_buf.data(), shape, perm, bit_width));
  EXPECT_EQ(in_place_buf, dst);
}

TEST(TflOpsInPlaceTransposeTest, SubByte2DTranspose1Bit) {
  // Shape: 8 rows x 8 cols, bit_width = 1
  const int64_t R = 8;
  const int64_t C = 8;
  const int bit_width = 1;
  const size_t total_elements = R * C;
  const size_t byte_count = (total_elements * bit_width + 7) / 8;

  std::vector<uint8_t> src(byte_count, 0);
  for (size_t i = 0; i < total_elements; ++i) {
    WriteSubByte(src.data(), i, static_cast<uint8_t>(i % 2), bit_width);
  }

  int64_t shape[] = {R, C};
  int64_t perm[] = {1, 0};
  std::vector<uint8_t> dst(byte_count, 0);
  EXPECT_TRUE(TransposeBuffer(src.data(), dst.data(), shape, perm, bit_width));

  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      uint8_t expected = ReadSubByte(src.data(), r * C + c, bit_width);
      uint8_t actual = ReadSubByte(dst.data(), c * R + r, bit_width);
      EXPECT_EQ(actual, expected) << "Mismatch at r=" << r << ", c=" << c;
    }
  }

  std::vector<uint8_t> in_place_buf = src;
  EXPECT_TRUE(InPlaceTranspose(in_place_buf.data(), shape, perm, bit_width));
  EXPECT_EQ(in_place_buf, dst);
}

TEST(TflOpsInPlaceTransposeTest, SubByte3DBatchTranspose) {
  // Shape: 2 batches x 4 rows x 6 cols, bit_width = 4, perm = [0, 2, 1]
  const int64_t B = 2;
  const int64_t R = 4;
  const int64_t C = 6;
  const int bit_width = 4;
  const size_t total_elements = B * R * C;
  const size_t byte_count = (total_elements * bit_width + 7) / 8;

  std::vector<uint8_t> src(byte_count, 0);
  for (size_t i = 0; i < total_elements; ++i) {
    WriteSubByte(src.data(), i, static_cast<uint8_t>(i % 16), bit_width);
  }

  int64_t shape[] = {B, R, C};
  int64_t perm[] = {0, 2, 1};
  std::vector<uint8_t> dst(byte_count, 0);
  EXPECT_TRUE(TransposeBuffer(src.data(), dst.data(), shape, perm, bit_width));

  for (int64_t b = 0; b < B; ++b) {
    for (int64_t r = 0; r < R; ++r) {
      for (int64_t c = 0; c < C; ++c) {
        size_t src_idx = (b * R + r) * C + c;
        size_t dst_idx = (b * C + c) * R + r;
        uint8_t expected = ReadSubByte(src.data(), src_idx, bit_width);
        uint8_t actual = ReadSubByte(dst.data(), dst_idx, bit_width);
        EXPECT_EQ(actual, expected)
            << "Mismatch at b=" << b << ", r=" << r << ", c=" << c;
      }
    }
  }

  std::vector<uint8_t> in_place_buf = src;
  EXPECT_TRUE(InPlaceTranspose(in_place_buf.data(), shape, perm, bit_width));
  EXPECT_EQ(in_place_buf, dst);
}

TEST(TflOpsInPlaceTransposeTest, SubByteNDGeneralTranspose) {
  // Shape: 2 x 3 x 4, bit_width = 4, perm = [2, 0, 1] -> output shape 4 x 2 x 3
  const int64_t shape[] = {2, 3, 4};
  const int64_t perm[] = {2, 0, 1};
  const int bit_width = 4;
  const size_t total_elements = 2 * 3 * 4;
  const size_t byte_count = (total_elements * bit_width + 7) / 8;

  std::vector<uint8_t> src(byte_count, 0);
  for (size_t i = 0; i < total_elements; ++i) {
    WriteSubByte(src.data(), i, static_cast<uint8_t>(i % 16), bit_width);
  }

  std::vector<uint8_t> dst(byte_count, 0);
  EXPECT_TRUE(TransposeBuffer(src.data(), dst.data(), shape, perm, bit_width));

  for (int64_t i = 0; i < 2; ++i) {
    for (int64_t j = 0; j < 3; ++j) {
      for (int64_t k = 0; k < 4; ++k) {
        size_t src_idx = (i * 3 + j) * 4 + k;
        size_t dst_idx = (k * 2 + i) * 3 + j;
        uint8_t expected = ReadSubByte(src.data(), src_idx, bit_width);
        uint8_t actual = ReadSubByte(dst.data(), dst_idx, bit_width);
        EXPECT_EQ(actual, expected)
            << "Mismatch at " << i << "," << j << "," << k;
      }
    }
  }

  std::vector<uint8_t> in_place_buf = src;
  EXPECT_TRUE(InPlaceTranspose(in_place_buf.data(), shape, perm, bit_width));
  EXPECT_EQ(in_place_buf, dst);
}

TEST(TflOpsInPlaceTransposeTest, StandardByteAlignedTranspose) {
  // Shape 3 x 5 of float (32-bit = 4 bytes)
  const int64_t R = 3;
  const int64_t C = 5;
  std::vector<float> src(R * C);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<float>(i * 1.5f);
  }

  int64_t shape[] = {R, C};
  int64_t perm[] = {1, 0};
  std::vector<float> dst(R * C);

  EXPECT_TRUE(
      TransposeBuffer(src.data(), dst.data(), shape, perm, /*bit_width=*/32));
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      EXPECT_EQ(dst[c * R + r], src[r * C + c]);
    }
  }

  std::vector<float> in_place_buf = src;
  EXPECT_TRUE(
      InPlaceTranspose(in_place_buf.data(), shape, perm, /*bit_width=*/32));
  EXPECT_EQ(in_place_buf, dst);
}

TEST(TflOpsInPlaceTransposeTest, NonStandardElementSizeTranspose) {
  // Element size 24 bits (3 bytes), e.g. RGB
  const int64_t R = 2;
  const int64_t C = 3;
  const int bit_width = 24;
  std::vector<uint8_t> src(R * C * 3);
  for (size_t i = 0; i < src.size(); ++i) {
    src[i] = static_cast<uint8_t>(i);
  }

  int64_t shape[] = {R, C};
  int64_t perm[] = {1, 0};
  std::vector<uint8_t> dst(R * C * 3);
  EXPECT_TRUE(TransposeBuffer(src.data(), dst.data(), shape, perm, bit_width));

  for (int64_t r = 0; r < R; ++r) {
    for (int64_t c = 0; c < C; ++c) {
      size_t src_offset = (r * C + c) * 3;
      size_t dst_offset = (c * R + r) * 3;
      EXPECT_EQ(dst[dst_offset], src[src_offset]);
      EXPECT_EQ(dst[dst_offset + 1], src[src_offset + 1]);
      EXPECT_EQ(dst[dst_offset + 2], src[src_offset + 2]);
    }
  }

  std::vector<uint8_t> in_place_buf = src;
  EXPECT_TRUE(InPlaceTranspose(in_place_buf.data(), shape, perm, bit_width));
  EXPECT_EQ(in_place_buf, dst);
}

}  // namespace
}  // namespace TFL
}  // namespace mlir
