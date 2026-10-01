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
#include "tensorflow/lite/delegates/xnnpack/moe_block_scale.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace tflite {
namespace xnnpack {
namespace {

using ::testing::ElementsAre;

TEST(ResolveBlockScaleLayoutTest, OneScalePerRowIsPerChannel) {
  const BlockScaleLayout layout = ResolveBlockScaleLayout(
      /*scale_elements=*/8, /*num_rows=*/8, /*input_channels=*/16);
  EXPECT_EQ(layout.groups_per_row, 1);
  EXPECT_EQ(layout.group_size, 16);
}

TEST(ResolveBlockScaleLayoutTest, SplitsInputAxisIntoEqualBlocks) {
  const BlockScaleLayout layout = ResolveBlockScaleLayout(
      /*scale_elements=*/32, /*num_rows=*/8, /*input_channels=*/64);
  EXPECT_EQ(layout.groups_per_row, 4);
  EXPECT_EQ(layout.group_size, 16);
}

// A scale per input channel is the finest blocking the layout can express.
TEST(ResolveBlockScaleLayoutTest, HandlesOneScalePerInputChannel) {
  const BlockScaleLayout layout = ResolveBlockScaleLayout(
      /*scale_elements=*/64, /*num_rows=*/8, /*input_channels=*/8);
  EXPECT_EQ(layout.groups_per_row, 8);
  EXPECT_EQ(layout.group_size, 1);
}

// More scales than input channels cannot describe a real blocking. The parser
// rejects such tensors, but the layout still has to stay in bounds rather than
// compute a zero group size and divide by it.
TEST(ResolveBlockScaleLayoutTest, ClampsGroupSizeWhenBlocksExceedChannels) {
  const BlockScaleLayout layout = ResolveBlockScaleLayout(
      /*scale_elements=*/64, /*num_rows=*/8, /*input_channels=*/4);
  EXPECT_EQ(layout.groups_per_row, 8);
  EXPECT_EQ(layout.group_size, 1);
}

TEST(ResolveBlockScaleLayoutTest, ToleratesZeroRows) {
  const BlockScaleLayout layout = ResolveBlockScaleLayout(
      /*scale_elements=*/0, /*num_rows=*/0, /*input_channels=*/16);
  EXPECT_EQ(layout.groups_per_row, 0);
  EXPECT_EQ(layout.group_size, 1);
}

TEST(BlockScaleIndexTest, PerChannelLayoutAlwaysUsesTheFirstScale) {
  const BlockScaleLayout layout = {/*groups_per_row=*/1, /*group_size=*/16};
  EXPECT_EQ(BlockScaleIndex(layout, 0), 0);
  EXPECT_EQ(BlockScaleIndex(layout, 15), 0);
}

TEST(BlockScaleIndexTest, AdvancesAtBlockBoundaries) {
  const BlockScaleLayout layout = {/*groups_per_row=*/4, /*group_size=*/4};
  std::vector<size_t> indices;
  indices.reserve(16);
  for (size_t in = 0; in < 16; ++in) {
    indices.push_back(BlockScaleIndex(layout, in));
  }
  EXPECT_THAT(indices,
              ElementsAre(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3));
}

// Reading past the last block would walk into the next row's scales, so the
// index saturates instead.
TEST(BlockScaleIndexTest, ClampsToTheLastBlock) {
  const BlockScaleLayout layout = {/*groups_per_row=*/4, /*group_size=*/4};
  EXPECT_EQ(BlockScaleIndex(layout, 16), 3);
  EXPECT_EQ(BlockScaleIndex(layout, 1000), 3);
}

TEST(BlockScaleIndexTest, ToleratesZeroGroupSize) {
  const BlockScaleLayout layout = {/*groups_per_row=*/4, /*group_size=*/0};
  EXPECT_EQ(BlockScaleIndex(layout, 7), 0);
}

// Two experts, two output channels, four input channels. Rows are interleaved
// as [out, expert], which is what makes the expert stride `num_experts`.
class DequantizeInt8Test : public ::testing::Test {
 protected:
  static constexpr size_t kNumExperts = 2;
  static constexpr size_t kOutputChannels = 2;
  static constexpr size_t kInputChannels = 4;

  // Row r holds the values {1, 2, 3, 4} offset by 10 * r, so a mis-indexed row
  // is obvious in the output.
  std::vector<int8_t> Weights() const {
    std::vector<int8_t> weights;
    for (size_t row = 0; row < kOutputChannels * kNumExperts; ++row) {
      for (size_t in = 0; in < kInputChannels; ++in) {
        weights.push_back(static_cast<int8_t>(10 * row + in + 1));
      }
    }
    return weights;
  }
};

TEST_F(DequantizeInt8Test, PerChannelScalesApplyToTheWholeRow) {
  const std::vector<int8_t> weights = Weights();
  // One scale per row; rows are [out0/e0, out0/e1, out1/e0, out1/e1].
  const std::vector<float> scales = {1.0f, 100.0f, 2.0f, 200.0f};
  std::vector<float> dst(kOutputChannels * kInputChannels, -1.0f);

  CopyAndDequantizeExpertWeightRowsInt8(
      weights.data(), scales.data(), scales.size(), kNumExperts, /*expert=*/0,
      kOutputChannels, kInputChannels, dst.data());

  // Expert 0 owns rows 0 and 2, scaled by 1.0 and 2.0.
  EXPECT_THAT(dst, ElementsAre(1.0f, 2.0f, 3.0f, 4.0f,        // row 0 * 1.0
                               42.0f, 44.0f, 46.0f, 48.0f));  // row 2 * 2.0
}

TEST_F(DequantizeInt8Test, BlockwiseScalesChangeHalfwayThroughTheRow) {
  const std::vector<int8_t> weights = Weights();
  // Two scales per row over four input channels, so a block is two channels
  // wide. Rows are [out0/e0, out0/e1, out1/e0, out1/e1].
  const std::vector<float> scales = {1.0f,  10.0f,    // row 0
                                     -1.0f, -10.0f,   // row 1 (other expert)
                                     2.0f,  20.0f,    // row 2
                                     -2.0f, -20.0f};  // row 3 (other expert)
  std::vector<float> dst(kOutputChannels * kInputChannels, -1.0f);

  CopyAndDequantizeExpertWeightRowsInt8(
      weights.data(), scales.data(), scales.size(), kNumExperts, /*expert=*/0,
      kOutputChannels, kInputChannels, dst.data());

  EXPECT_THAT(dst, ElementsAre(
                       // row 0 = {1,2,3,4}, scales {1, 1, 10, 10}
                       1.0f, 2.0f, 30.0f, 40.0f,
                       // row 2 = {21,22,23,24}, scales {2, 2, 20, 20}
                       42.0f, 44.0f, 460.0f, 480.0f));
}

// The second expert must read its own interleaved rows and its own scales.
TEST_F(DequantizeInt8Test, BlockwiseScalesAreSelectedPerExpert) {
  const std::vector<int8_t> weights = Weights();
  const std::vector<float> scales = {1.0f, 10.0f, -1.0f, -10.0f,
                                     2.0f, 20.0f, -2.0f, -20.0f};
  std::vector<float> dst(kOutputChannels * kInputChannels, 0.0f);

  CopyAndDequantizeExpertWeightRowsInt8(
      weights.data(), scales.data(), scales.size(), kNumExperts, /*expert=*/1,
      kOutputChannels, kInputChannels, dst.data());

  EXPECT_THAT(dst, ElementsAre(
                       // row 1 = {11,12,13,14}, scales {-1, -1, -10, -10}
                       -11.0f, -12.0f, -130.0f, -140.0f,
                       // row 3 = {31,32,33,34}, scales {-2, -2, -20, -20}
                       -62.0f, -64.0f, -660.0f, -680.0f));
}

// One scale per input channel is the limiting case of blocking.
TEST_F(DequantizeInt8Test, SupportsOneScalePerInputChannel) {
  const std::vector<int8_t> weights = Weights();
  std::vector<float> scales(kOutputChannels * kNumExperts * kInputChannels,
                            0.0f);
  // Only expert 0's rows (0 and 2) matter here.
  for (size_t i = 0; i < kInputChannels; ++i) {
    scales[0 * kInputChannels + i] = static_cast<float>(i + 1);
    scales[2 * kInputChannels + i] = static_cast<float>(i + 1);
  }
  std::vector<float> dst(kOutputChannels * kInputChannels, 0.0f);

  CopyAndDequantizeExpertWeightRowsInt8(
      weights.data(), scales.data(), scales.size(), kNumExperts, /*expert=*/0,
      kOutputChannels, kInputChannels, dst.data());

  EXPECT_THAT(dst, ElementsAre(
                       // row 0 = {1,2,3,4} times {1,2,3,4}
                       1.0f, 4.0f, 9.0f, 16.0f,
                       // row 2 = {21,22,23,24} times {1,2,3,4}
                       21.0f, 44.0f, 69.0f, 96.0f));
}

class DequantizeInt4Test : public ::testing::Test {
 protected:
  static constexpr size_t kNumExperts = 2;
  static constexpr size_t kOutputChannels = 1;
  static constexpr size_t kInputChannels = 4;

  // Two nibbles per byte, low nibble first. Row 0 holds {1, 2, 3, 4} and row 1
  // holds {-1, -2, -3, -4}.
  std::vector<int8_t> PackedWeights() const {
    return {
        static_cast<int8_t>(0x21),
        static_cast<int8_t>(0x43),  // row 0
        static_cast<int8_t>(0xEF),
        static_cast<int8_t>(0xCD),  // row 1
    };
  }
};

TEST_F(DequantizeInt4Test, UnpacksNibblesWithPerChannelScale) {
  const std::vector<int8_t> weights = PackedWeights();
  const std::vector<float> scales = {2.0f, 3.0f};
  std::vector<float> dst(kInputChannels, 0.0f);

  CopyAndDequantizeExpertWeightRowsInt4(
      weights.data(), scales.data(), scales.size(), kNumExperts, /*expert=*/0,
      kOutputChannels, kInputChannels, dst.data());

  EXPECT_THAT(dst, ElementsAre(2.0f, 4.0f, 6.0f, 8.0f));
}

TEST_F(DequantizeInt4Test, AppliesBlockwiseScalesToNibbles) {
  const std::vector<int8_t> weights = PackedWeights();
  // Two scales per row over four input channels.
  const std::vector<float> scales = {1.0f, 10.0f,   // row 0
                                     5.0f, 50.0f};  // row 1
  std::vector<float> dst(kInputChannels, 0.0f);

  CopyAndDequantizeExpertWeightRowsInt4(
      weights.data(), scales.data(), scales.size(), kNumExperts, /*expert=*/1,
      kOutputChannels, kInputChannels, dst.data());

  // Row 1 = {-1, -2, -3, -4} with scales {5, 5, 50, 50}.
  EXPECT_THAT(dst, ElementsAre(-5.0f, -10.0f, -150.0f, -200.0f));
}

}  // namespace
}  // namespace xnnpack
}  // namespace tflite
