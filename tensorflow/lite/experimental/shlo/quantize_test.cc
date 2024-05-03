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

#include "tensorflow/lite/experimental/shlo/quantize.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/experimental/shlo/data_type.h"

namespace shlo_ref {
namespace {

using ::testing::Eq;

template <DataType storage_type, DataType expressed_type>
struct TestPair {
  using StorageT = StorageType<storage_type>;
  using ExpressedT = StorageType<expressed_type>;

  static constexpr DataType kStorageType = storage_type;
  static constexpr DataType kExpressedType = expressed_type;
};

template <typename T>
class QuantizationTypeTest : public ::testing::Test {};

// TODO(rjascani): Including F16 in these tests can cause linker failures
// because of https://github.com/llvm/llvm-project/issues/77786. Once that issue
// is resolved, we should add F16 back to the list.
using TestTypes = ::testing::Types<TestPair<DataType::kSI4, DataType::kBF16>,
                                   TestPair<DataType::kSI4, DataType::kF32>,
                                   TestPair<DataType::kSI8, DataType::kBF16>,
                                   TestPair<DataType::kSI8, DataType::kF32>,
                                   TestPair<DataType::kSI16, DataType::kBF16>,
                                   TestPair<DataType::kSI16, DataType::kF32>,
                                   TestPair<DataType::kSI32, DataType::kBF16>,
                                   TestPair<DataType::kSI32, DataType::kF32>>;

TYPED_TEST_SUITE(QuantizationTypeTest, TestTypes);

TYPED_TEST(QuantizationTypeTest, Dequantize) {
  typename TypeParam::StorageT quantized_value{5};
  typename TypeParam::StorageT zero_point{3};
  typename TypeParam::ExpressedT scale{.5f};
  typename TypeParam::ExpressedT expected_value{1.0f};

  EXPECT_THAT(Dequantize(quantized_value, zero_point, scale),
              Eq(expected_value));
}

TYPED_TEST(QuantizationTypeTest, Quantize) {
  typename TypeParam::ExpressedT expressed_value{1.0f};
  typename TypeParam::StorageT zero_point{3};
  typename TypeParam::ExpressedT scale_inv{2.0f};
  typename TypeParam::StorageT expected_value{5};

  EXPECT_THAT((Quantize<TypeParam::kStorageType, TypeParam::kExpressedType>(
                  expressed_value, zero_point, scale_inv)),
              Eq(expected_value));
}

TEST(QuantizeTest, QuantizedValueClamped) {
  using StorageT = StorageType<DataType::kSI4>;
  using ExpressedT = StorageType<DataType::kF32>;

  ExpressedT expressed_value = Storage<DataType::kF32>::kMaxValue - 1;
  StorageT zero_point = 5;
  ExpressedT scale_inv = 2;
  StorageT expected_value = Storage<DataType::kSI4>::kMaxValue;

  EXPECT_THAT((Quantize<DataType::kSI4, DataType::kF32>(expressed_value,
                                                        zero_point, scale_inv)),
              Eq(expected_value));
}

TEST(QuantizeTest, SI4NegativeValue) {
  using StorageT = StorageType<DataType::kSI4>;
  using ExpressedT = StorageType<DataType::kF32>;

  StorageT value = -8;
  StorageT zero_point = 0;
  ExpressedT scale = 1;
  ExpressedT expected_value = -8.0f;

  EXPECT_THAT((Dequantize(value, zero_point, scale)), Eq(expected_value));
  EXPECT_THAT((Quantize<DataType::kSI4, DataType::kF32>(expected_value,
                                                        zero_point, 1 / scale)),
              Eq(value));
}

TEST(QuantizeTest, QuantizedValueRoundDown) {
  using StorageT = StorageType<DataType::kSI8>;
  using ExpressedT = StorageType<DataType::kF32>;

  ExpressedT expressed_value = 2.2f;
  StorageT zero_point = 5;
  ExpressedT scale_inv = 2;
  StorageT expected_value = 9;

  EXPECT_THAT((Quantize<DataType::kSI8, DataType::kF32>(expressed_value,
                                                        zero_point, scale_inv)),
              Eq(expected_value));
}

TEST(QuantizeTest, QuantizedValueRoundUp) {
  using StorageT = StorageType<DataType::kSI8>;
  using ExpressedT = StorageType<DataType::kF32>;

  ExpressedT expressed_value = 2.4f;
  StorageT zero_point = 5;
  ExpressedT scale_inv = 2;
  StorageT expected_value = 10;

  EXPECT_THAT((Quantize<DataType::kSI8, DataType::kF32>(expressed_value,
                                                        zero_point, scale_inv)),
              Eq(expected_value));
}

}  // namespace
}  // namespace shlo_ref
