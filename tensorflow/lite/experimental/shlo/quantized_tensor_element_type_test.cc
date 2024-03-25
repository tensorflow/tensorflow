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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {
namespace {

using testing::Each;
using testing::ElementsAre;
using testing::Eq;

template <DataType storage_type, DataType expressed_type>
struct TestPair {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  static constexpr DataType kStorageType = storage_type;
  static constexpr DataType kExpressedType = expressed_type;
};

template <typename T>
class QuantizedTensorElementTypeTest : public ::testing::Test {};

using TestTypes = ::testing::Types<TestPair<DataType::kSI4, DataType::kBF16>,
                                   TestPair<DataType::kSI4, DataType::kF16>,
                                   TestPair<DataType::kSI4, DataType::kF32>,
                                   TestPair<DataType::kSI8, DataType::kBF16>,
                                   TestPair<DataType::kSI8, DataType::kF16>,
                                   TestPair<DataType::kSI8, DataType::kF32>,
                                   TestPair<DataType::kSI16, DataType::kBF16>,
                                   TestPair<DataType::kSI16, DataType::kF16>,
                                   TestPair<DataType::kSI16, DataType::kF32>,
                                   TestPair<DataType::kSI32, DataType::kBF16>,
                                   TestPair<DataType::kSI32, DataType::kF16>,
                                   TestPair<DataType::kSI32, DataType::kF32>>;

TYPED_TEST_SUITE(QuantizedTensorElementTypeTest, TestTypes);

TYPED_TEST(QuantizedTensorElementTypeTest, PerTensor) {
  typename TypeParam::ExpressedT scale{.5f};
  typename TypeParam::StorageT zero_point{3};

  auto element =
      QuantizedTensorElementType::PerTensor<TypeParam::kStorageType,
                                            TypeParam::kExpressedType>(
          scale, zero_point);

  EXPECT_THAT(element.StorageType(), Eq(TypeParam::kStorageType));
  EXPECT_THAT(element.ExpressedType(), Eq(TypeParam::kExpressedType));
  EXPECT_THAT(element.IsPerTensorQuantized(), Eq(true));
  EXPECT_THAT(element.IsPerAxisQuantized(), Eq(false));
  EXPECT_THAT(element.template Scales<TypeParam::kExpressedType>(),
              ElementsAre(.5f));
  EXPECT_THAT(element.template ZeroPoints<TypeParam::kStorageType>(),
              ElementsAre(3));
}

TYPED_TEST(QuantizedTensorElementTypeTest, PerAxis) {
  using ExpressedT = typename TypeParam::ExpressedT;
  using StorageT = typename TypeParam::StorageT;
  std::array scales = {ExpressedT{.5f}, ExpressedT{.6f}, ExpressedT{.2f}};
  std::array zero_points = {StorageT{3}, StorageT{1}, StorageT{2}};

  auto element = QuantizedTensorElementType::PerAxis<TypeParam::kStorageType,
                                                     TypeParam::kExpressedType>(
      absl::MakeConstSpan(scales), absl::MakeConstSpan(zero_points), 3u);

  EXPECT_THAT(element.StorageType(), Eq(TypeParam::kStorageType));
  EXPECT_THAT(element.ExpressedType(), Eq(TypeParam::kExpressedType));
  EXPECT_THAT(element.IsPerTensorQuantized(), Eq(false));
  EXPECT_THAT(element.IsPerAxisQuantized(), Eq(true));
  EXPECT_THAT(element.QuantizedDimension(), Eq(3));
  EXPECT_THAT(element.template Scales<TypeParam::kExpressedType>(),
              ElementsAre(ExpressedT{.5f}, ExpressedT{.6f}, ExpressedT{.2f}));
  EXPECT_THAT(element.template ZeroPoints<TypeParam::kStorageType>(),
              ElementsAre(3, 1, 2));
}

TYPED_TEST(QuantizedTensorElementTypeTest, BaselineTypeForPerAxisWorks) {
  using ExpressedT = typename TypeParam::ExpressedT;
  ExpressedT scales[3] = {ExpressedT(0.5f), ExpressedT(0.6f), ExpressedT(0.2f)};
  typename TypeParam::StorageT zero_points[3] = {3, 1, 2};
  const auto element =
      QuantizedTensorElementType::PerAxis<TypeParam::kStorageType,
                                          TypeParam::kExpressedType>(
          absl::MakeConstSpan(scales), absl::MakeConstSpan(zero_points),
          /*quantized_dimension=*/3u);
  const auto baseline = BaselineType(element);

  EXPECT_EQ(baseline.StorageType(), element.StorageType());
  EXPECT_EQ(baseline.ExpressedType(), element.ExpressedType());
  EXPECT_EQ(baseline.IsPerTensorQuantized(), element.IsPerTensorQuantized());
  EXPECT_EQ(baseline.IsPerAxisQuantized(), element.IsPerAxisQuantized());
  EXPECT_EQ(baseline.QuantizedDimension(), element.QuantizedDimension());
  EXPECT_THAT(baseline.template Scales<TypeParam::kExpressedType>().size(),
              element.template Scales<TypeParam::kExpressedType>().size());
  EXPECT_THAT(baseline.template Scales<TypeParam::kExpressedType>(), Each(1.f));
  EXPECT_THAT(baseline.template ZeroPoints<TypeParam::kStorageType>().size(),
              element.template ZeroPoints<TypeParam::kStorageType>().size());
  EXPECT_THAT(baseline.template ZeroPoints<TypeParam::kStorageType>(), Each(0));
}

TYPED_TEST(QuantizedTensorElementTypeTest, BaselineTypeForPerTensorWorks) {
  typename TypeParam::ExpressedT scale(0.5f);
  typename TypeParam::StorageT zero_point = 3;

  const auto element =
      QuantizedTensorElementType::PerTensor<TypeParam::kStorageType,
                                            TypeParam::kExpressedType>(
          scale, zero_point);
  const auto baseline = BaselineType(element);

  EXPECT_EQ(baseline.StorageType(), element.StorageType());
  EXPECT_EQ(baseline.ExpressedType(), element.ExpressedType());
  EXPECT_EQ(baseline.IsPerTensorQuantized(), element.IsPerTensorQuantized());
  EXPECT_EQ(baseline.IsPerAxisQuantized(), element.IsPerAxisQuantized());
  EXPECT_THAT(baseline.template Scales<TypeParam::kExpressedType>().size(),
              element.template Scales<TypeParam::kExpressedType>().size());
  EXPECT_THAT(baseline.template Scales<TypeParam::kExpressedType>(), Each(1.f));
  EXPECT_THAT(baseline.template ZeroPoints<TypeParam::kStorageType>().size(),
              element.template ZeroPoints<TypeParam::kStorageType>().size());
  EXPECT_THAT(baseline.template ZeroPoints<TypeParam::kStorageType>(), Each(0));
}

}  // namespace
}  // namespace shlo_ref
