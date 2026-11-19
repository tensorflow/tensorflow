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
#include "tensorflow/compiler/mlir/lite/tools/optimize/quantization_utils.h"

#include <cstdint>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace mlir {
namespace TFL {
namespace {

using ::testing::ElementsAre;

TEST(SymmetricQuantizeFloatsToInt16Test, RoundsAndClamps) {
  const std::vector<float> data = {0.0f, 1.0f, -1.0f, 2.6f, 1e9f, -1e9f};
  EXPECT_THAT(SymmetricQuantizeFloatsToInt16(data.data(), data.size(),
                                             /*scaling_factor=*/0.5f),
              ElementsAre(0, 2, -2, 5, 32767, -32767));
}

TEST(SymmetricQuantizeFloatsToInt16Test, ZeroScaleGivesZeros) {
  const std::vector<float> data = {1.0f, -3.0f};
  EXPECT_THAT(SymmetricQuantizeFloatsToInt16(data.data(), data.size(),
                                             /*scaling_factor=*/0.0f),
              ElementsAre(0, 0));
}

TEST(SymmetricPerChannelQuantizeValuesTest, LeadingChannelDim) {
  // Shape [2, 2], channel dimension 0.
  const std::vector<float> input = {1.4f, -100.4f, 0.25f, -0.26f};
  std::vector<int8_t> output(input.size());
  SymmetricPerChannelQuantizeValues(input.data(), /*scales_inv=*/{1.0f, 10.0f},
                                    /*dimension=*/{2, 2},
                                    /*channel_dim_index=*/0, &output);
  // Values are rounded half away from zero.
  EXPECT_THAT(output, ElementsAre(1, -100, 3, -3));
}

TEST(SymmetricPerChannelQuantizeValuesTest, TrailingChannelDim) {
  // Shape [2, 2], channel dimension 1.
  const std::vector<float> input = {1.0f, 1.0f, 2.0f, 2.0f};
  std::vector<int8_t> output(input.size());
  SymmetricPerChannelQuantizeValues(input.data(), /*scales_inv=*/{1.0f, 10.0f},
                                    /*dimension=*/{2, 2},
                                    /*channel_dim_index=*/1, &output);
  EXPECT_THAT(output, ElementsAre(1, 10, 2, 20));
}

TEST(SymmetricPerChannelQuantizeValuesTest, OneDimension) {
  const std::vector<float> input = {1.0f, 1.0f};
  std::vector<int8_t> output(input.size());
  SymmetricPerChannelQuantizeValues(input.data(), /*scales_inv=*/{2.0f, 3.0f},
                                    /*dimension=*/{2},
                                    /*channel_dim_index=*/0, &output);
  EXPECT_THAT(output, ElementsAre(2, 3));
}

TEST(SymmetricPerChannelQuantizeValuesDeathTest, TooManyDimensions) {
  const std::vector<float> input(1, 1.0f);
  std::vector<int8_t> output(input.size());
  EXPECT_DEATH(
      SymmetricPerChannelQuantizeValues(input.data(), /*scales_inv=*/{1.0f},
                                        /*dimension=*/{1, 1, 1, 1, 1},
                                        /*channel_dim_index=*/0, &output),
      "dimensions are supported");
}

TEST(SymmetricPerChannelQuantizeValuesDeathTest, ChannelDimOutOfRange) {
  const std::vector<float> input(2, 1.0f);
  std::vector<int8_t> output(input.size());
  EXPECT_DEATH(SymmetricPerChannelQuantizeValues(
                   input.data(), /*scales_inv=*/{1.0f, 1.0f},
                   /*dimension=*/{2}, /*channel_dim_index=*/1, &output),
               "Invalid channel dimension");
  EXPECT_DEATH(SymmetricPerChannelQuantizeValues(
                   input.data(), /*scales_inv=*/{1.0f, 1.0f},
                   /*dimension=*/{2}, /*channel_dim_index=*/-1, &output),
               "Invalid channel dimension");
}

TEST(SymmetricPerChannelQuantizeValuesDeathTest, TooFewScales) {
  const std::vector<float> input(2, 1.0f);
  std::vector<int8_t> output(input.size());
  EXPECT_DEATH(SymmetricPerChannelQuantizeValues(
                   input.data(), /*scales_inv=*/{1.0f}, /*dimension=*/{2},
                   /*channel_dim_index=*/0, &output),
               "one scale per channel");
}

TEST(SymmetricBiasQuantizeTest, PerLayerInt32) {
  const std::vector<float> data = {1.0f, -1.25f, 3e9f, -3e9f};
  constexpr int32_t kMax = std::numeric_limits<int32_t>::max();
  EXPECT_THAT(SymmetricBiasQuantize<int32_t>(data.data(), data.size(),
                                             /*scales=*/{0.5f}),
              ElementsAre(2, -3, kMax, -kMax));
}

TEST(SymmetricBiasQuantizeTest, PerChannelInt32WithZeroScale) {
  const std::vector<float> data = {1.0f, 1.0f};
  EXPECT_THAT(SymmetricBiasQuantize<int32_t>(data.data(), data.size(),
                                             /*scales=*/{0.5f, 0.0f}),
              ElementsAre(2, 0));
}

TEST(SymmetricBiasQuantizeTest, NanBecomesZero) {
  const std::vector<float> data = {std::numeric_limits<float>::quiet_NaN()};
  EXPECT_THAT(SymmetricBiasQuantize<int32_t>(data.data(), data.size(),
                                             /*scales=*/{1.0f}),
              ElementsAre(0));
}

TEST(SymmetricBiasQuantizeTest, Int64) {
  // 2^40 is exactly representable as a float and doesn't fit in int32.
  const std::vector<float> data = {1099511627776.0f};
  EXPECT_THAT(SymmetricBiasQuantize<int64_t>(data.data(), data.size(),
                                             /*scales=*/{1.0f}),
              ElementsAre(int64_t{1099511627776}));
}

TEST(SymmetricBiasQuantizeDeathTest, EmptyScales) {
  const std::vector<float> data = {1.0f};
  EXPECT_DEATH(SymmetricBiasQuantize<int32_t>(data.data(), data.size(),
                                              /*scales=*/{}),
               "Scales must not be empty");
}

TEST(SymmetricBiasQuantizeDeathTest, TooFewPerChannelScales) {
  const std::vector<float> data = {1.0f, 1.0f, 1.0f};
  EXPECT_DEATH(SymmetricBiasQuantize<int32_t>(data.data(), data.size(),
                                              /*scales=*/{1.0f, 1.0f}),
               "Expected 1 or at least 3 scales");
}

}  // namespace
}  // namespace TFL
}  // namespace mlir
