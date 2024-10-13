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

#include "tensorflow/lite/experimental/shlo/quantized_tensor_element_type.h"

#include <array>
#include <cstdint>
#include <variant>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {
namespace {

using testing::Each;
using testing::ElementsAreArray;
using testing::FloatEq;
using testing::Pointwise;

TEST(Quantization, IsValidQuantizationTypePairWorks) {
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kSI32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kBF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kI1, DataType::kF32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kSI32));
  EXPECT_TRUE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kBF16));
  EXPECT_TRUE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kF16));
  EXPECT_TRUE(IsValidQuantizationTypePair(DataType::kSI4, DataType::kF32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kSI32));
  EXPECT_TRUE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kBF16));
  EXPECT_TRUE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kF16));
  EXPECT_TRUE(IsValidQuantizationTypePair(DataType::kSI8, DataType::kF32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kSI32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kBF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kF16));
  EXPECT_TRUE(IsValidQuantizationTypePair(DataType::kSI16, DataType::kF32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kSI32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kBF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kSI32, DataType::kF32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kSI32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kBF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kBF16, DataType::kF32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kSI32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kBF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF16, DataType::kF32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kI1));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kSI4));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kSI8));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kSI16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kSI32));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kBF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kF16));
  EXPECT_FALSE(IsValidQuantizationTypePair(DataType::kF32, DataType::kF32));
}

struct QuantizationPair {
  DataType storage_type;
  DataType expressed_type;
};

std::vector<QuantizationPair> ValidQuantizationTypePairs() {
  return {QuantizationPair{.storage_type = DataType::kSI4,
                           .expressed_type = DataType::kBF16},
          QuantizationPair{.storage_type = DataType::kSI4,
                           .expressed_type = DataType::kF16},
          QuantizationPair{.storage_type = DataType::kSI4,
                           .expressed_type = DataType::kF32},
          QuantizationPair{.storage_type = DataType::kSI8,
                           .expressed_type = DataType::kBF16},
          QuantizationPair{.storage_type = DataType::kSI8,
                           .expressed_type = DataType::kF16},
          QuantizationPair{.storage_type = DataType::kSI8,
                           .expressed_type = DataType::kF32},
          QuantizationPair{.storage_type = DataType::kSI16,
                           .expressed_type = DataType::kF32}};
}

struct PerTensorTest : testing::TestWithParam<QuantizationPair> {
  // NOLINTNEXTLINE: Using function naming for functors.
  static constexpr auto ExtractValueAsInt = [](auto v) {
    return static_cast<int32_t>(v);
  };
  // NOLINTNEXTLINE: Using function naming for functors.
  static constexpr auto ExtractValueAsFloat = [](auto v) {
    return static_cast<float>(v);
  };
};

TEST_P(PerTensorTest, BuildPerTensorWorks) {
  const QuantizationPair& config = GetParam();
  QuantizedElementTypePerTensor type(config.storage_type, 1,
                                     config.expressed_type, 2.5);

  EXPECT_EQ(type.StorageType(), config.storage_type);
  EXPECT_EQ(type.ExpressedType(), config.expressed_type);
  EXPECT_EQ(std::visit(ExtractValueAsInt, type.ZeroPoint()), 1);
  EXPECT_THAT(std::visit(ExtractValueAsFloat, type.Scale()), FloatEq(2.5));
}

TEST_P(PerTensorTest, BaselineTypeWorks) {
  float scale = 0.5f;
  int32_t zero_point = 3;

  const QuantizationPair& config = GetParam();
  QuantizedElementTypePerTensor element(config.storage_type, zero_point,
                                        config.expressed_type, scale);
  const auto baseline = BaselineType(element);

  EXPECT_EQ(baseline.StorageType(), element.StorageType());
  EXPECT_EQ(baseline.ExpressedType(), element.ExpressedType());
  EXPECT_EQ(std::visit(ExtractValueAsInt, baseline.ZeroPoint()), 0);
  EXPECT_THAT(std::visit(ExtractValueAsFloat, baseline.Scale()), FloatEq(1));
}

INSTANTIATE_TEST_SUITE_P(PerTensor, PerTensorTest,
                         testing::ValuesIn(ValidQuantizationTypePairs()));

struct PerAxisTest : testing::TestWithParam<QuantizationPair> {
  // NOLINTNEXTLINE: Using function naming for functors.
  static constexpr auto ExtractValueAsInt = [](auto v) {
    return std::vector<int32_t>(v.begin(), v.end());
  };
  // NOLINTNEXTLINE: Using function naming for functors.
  static constexpr auto ExtractValueAsFloat = [](auto v) {
    return std::vector<float>(v.begin(), v.end());
  };
};

TEST_P(PerAxisTest, BuildPerAxisWorks) {
  const QuantizationPair& config = GetParam();
  const std::vector<int32_t> ref_zero_points{1, 2, 3};
  const std::vector<float> ref_scales{1.5, 2.5, 3.5};

  QuantizedElementTypePerAxis type(config.storage_type, ref_zero_points,
                                   config.expressed_type, ref_scales,
                                   /*quantized_dimension=*/1);

  EXPECT_EQ(type.StorageType(), config.storage_type);
  EXPECT_EQ(type.ExpressedType(), config.expressed_type);
  EXPECT_THAT(std::visit(ExtractValueAsInt, type.ZeroPoints()),
              ElementsAreArray(ref_zero_points));
  EXPECT_THAT(std::visit(ExtractValueAsFloat, type.Scales()),
              Pointwise(FloatEq(), ref_scales));
}

TEST_P(PerAxisTest, BaselineTypeWorks) {
  const QuantizationPair& config = GetParam();
  float scales[3] = {0.5f, 0.6f, 0.2f};
  int32_t zero_points[3] = {3, 1, 2};
  const QuantizedElementTypePerAxis element(config.storage_type, scales,
                                            config.expressed_type, zero_points,
                                            /*quantized_dimension=*/3u);
  const auto baseline = BaselineType(element);

  const auto extracted_zero_points =
      std::visit(ExtractValueAsInt, baseline.ZeroPoints());
  const auto extracted_scales =
      std::visit(ExtractValueAsFloat, baseline.Scales());

  EXPECT_EQ(baseline.StorageType(), element.StorageType());
  EXPECT_EQ(baseline.ExpressedType(), element.ExpressedType());
  EXPECT_EQ(baseline.QuantizedDimension(), element.QuantizedDimension());

  EXPECT_THAT(extracted_zero_points, Each(0));
  EXPECT_THAT(extracted_zero_points.size(), std::size(zero_points));
  EXPECT_THAT(extracted_scales, Each(FloatEq(1.0f)));
  EXPECT_THAT(extracted_scales.size(), std::size(scales));
}

INSTANTIATE_TEST_SUITE_P(PerAxis, PerAxisTest,
                         testing::ValuesIn(ValidQuantizationTypePairs()));

}  // namespace
}  // namespace shlo_ref
