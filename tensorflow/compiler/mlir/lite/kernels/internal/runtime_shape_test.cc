/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
// This file is the MLIR copy of runtime_shape as part of the effort to
// decouple TFLite from MLIR.
// LINT.IfChange

#include "tensorflow/compiler/mlir/lite/kernels/internal/runtime_shape.h"

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/types/span.h"

using testing::Each;
using testing::ElementsAreArray;

namespace mlir {
namespace {

constexpr int kSmallSize = RuntimeShape::kMaxSmallSize;
constexpr int kBigSize = RuntimeShape::kMaxSmallSize + 1;

std::vector<int32_t> IotaVector(int size, int start = 0) {
  std::vector<int32_t> vec(size);
  absl::c_iota(vec, start);
  return vec;
}

absl::Span<const int32_t> AsSpan(const RuntimeShape& shape) {
  return absl::Span<const int32_t>(shape.DimsData(), shape.DimensionsCount());
}

class RuntimeShapeTest : public testing::TestWithParam<int> {};

TEST(RuntimeShapeTest, TestDefaultConstructor) {
  const RuntimeShape shape;
  EXPECT_EQ(shape.DimensionsCount(), 0);
}

TEST_P(RuntimeShapeTest, TestConstructorWithSize) {
  const int size = GetParam();
  const RuntimeShape shape(size);
  EXPECT_EQ(shape.DimensionsCount(), size);
}

TEST_P(RuntimeShapeTest, TestConstructorWithSizeAndDefaultValue) {
  const int size = GetParam();
  const RuntimeShape shape(size, 34);
  EXPECT_EQ(shape.DimensionsCount(), size);
  EXPECT_THAT(AsSpan(shape), Each(34));
}

TEST_P(RuntimeShapeTest, TestConstructorFromCArray) {
  const int size = GetParam();
  const std::vector<int32_t> src = IotaVector(size);
  const RuntimeShape shape(size, src.data());
  EXPECT_EQ(shape.DimensionsCount(), size);
  EXPECT_THAT(AsSpan(shape), ElementsAreArray(src));
}

TEST(RuntimeShapeTest, TestConstructorFromSmallInitList) {
  std::initializer_list<int> init{1, 2, 3};
  // Ensure we are testing a small initializer list.
  ASSERT_LE(init.size(), RuntimeShape::kMaxSmallSize);
  const RuntimeShape shape(init);
  EXPECT_EQ(shape.DimensionsCount(), init.size());
  EXPECT_THAT(AsSpan(shape), ElementsAreArray(init));
}

TEST(RuntimeShapeTest, TestConstructorFromBigInitList) {
  std::initializer_list<int> init{1, 2, 3, 4, 5, 6, 7, 8, 9};
  // Ensure we are testing a big initializer list.
  ASSERT_GT(init.size(), RuntimeShape::kMaxSmallSize);
  const RuntimeShape shape(init);
  EXPECT_EQ(shape.DimensionsCount(), init.size());
  EXPECT_THAT(AsSpan(shape), ElementsAreArray(init));
}

TEST_P(RuntimeShapeTest, TestCopyConstructorFromShape) {
  const int size = GetParam();
  const RuntimeShape src(size, 34);
  const RuntimeShape dst(src);
  EXPECT_EQ(dst.DimensionsCount(), src.DimensionsCount());
  EXPECT_THAT(AsSpan(dst), ElementsAreArray(AsSpan(src)));
}

TEST_P(RuntimeShapeTest, TestEqualityOperator) {
  const int size = GetParam();
  const RuntimeShape shape1(size, 34);
  const RuntimeShape shape2(size, 34);
  EXPECT_TRUE(shape1 == shape2);
  EXPECT_FALSE(shape1 != shape2);
}

TEST_P(RuntimeShapeTest, TestEqualityOperatorDifferentSizes) {
  const int size = GetParam();
  const RuntimeShape shape1(size, 34);
  const RuntimeShape shape2(size + 1, 34);
  EXPECT_FALSE(shape1 == shape2);
  EXPECT_TRUE(shape1 != shape2);
}

TEST_P(RuntimeShapeTest, TestEqualityOperatorDifferentValues) {
  const int size = GetParam();
  const RuntimeShape shape1(size, 34);
  const RuntimeShape shape2(size, 43);
  EXPECT_FALSE(shape1 == shape2);
  EXPECT_TRUE(shape1 != shape2);
}

TEST_P(RuntimeShapeTest, TestSetterGetter) {
  const int size = GetParam();
  RuntimeShape shape(size);
  for (int i = 0; i < size; ++i) {
    shape.SetDim(i, i);
    EXPECT_EQ(shape.Dims(i), i);
  }
  EXPECT_THAT(AsSpan(shape), ElementsAreArray(IotaVector(size)));
}

TEST(RuntimeShapeTest, TestResizeSmallSmall) {
  ASSERT_GE(kSmallSize, 1);
  RuntimeShape shape(kSmallSize - 1, 23);
  shape.Resize(kSmallSize);
  EXPECT_EQ(shape.DimensionsCount(), kSmallSize);
  EXPECT_THAT(absl::Span<const int32_t>(shape.DimsData(), kSmallSize - 1),
              Each(23));
}

TEST(RuntimeShapeTest, TestResizeSmallBig) {
  RuntimeShape shape(kSmallSize, 23);
  shape.Resize(kBigSize);
  EXPECT_EQ(shape.DimensionsCount(), kBigSize);
  EXPECT_THAT(absl::Span<const int32_t>(shape.DimsData(), kSmallSize),
              Each(23));
}

TEST(RuntimeShapeTest, TestResizeBigSmall) {
  RuntimeShape shape(kBigSize, 23);
  shape.Resize(kSmallSize);
  EXPECT_EQ(shape.DimensionsCount(), kSmallSize);
  EXPECT_THAT(absl::Span<const int32_t>(shape.DimsData(), kSmallSize),
              Each(23));
}

TEST(RuntimeShapeTest, TestResizeDownBigBig) {
  RuntimeShape shape(kBigSize + 3, 23);
  shape.Resize(kBigSize);
  EXPECT_EQ(shape.DimensionsCount(), kBigSize);
  EXPECT_THAT(absl::Span<const int32_t>(shape.DimsData(), kBigSize), Each(23));
}

TEST(RuntimeShapeTest, TestResizeUpBigBig) {
  RuntimeShape shape(kBigSize, 23);
  shape.Resize(kBigSize + 1);
  EXPECT_EQ(shape.DimensionsCount(), kBigSize + 1);
  EXPECT_THAT(absl::Span<const int32_t>(shape.DimsData(), kBigSize), Each(23));
}

TEST_P(RuntimeShapeTest, TestReplaceWith) {
  static_assert(
      RuntimeShape::kMaxSmallSize > 2,
      "kMaxSmallSize should be greater than 2 for this test to work.");
  const int size = GetParam();
  for (const int offset : {-2, 2}) {
    const std::vector<int32_t> src =
        IotaVector(offset + RuntimeShape::kMaxSmallSize);
    RuntimeShape shape(size);
    shape.ReplaceWith(src.size(), src.data());
    EXPECT_EQ(shape.DimensionsCount(), src.size());
    EXPECT_THAT(AsSpan(shape), testing::ElementsAreArray(src));
  }
}

TEST_P(RuntimeShapeTest, TestBuildFrom) {
  const int size = GetParam();
  const std::vector<int32_t> src = IotaVector(size);
  RuntimeShape shape;
  shape.BuildFrom(src);
  EXPECT_EQ(shape.DimensionsCount(), src.size());
  EXPECT_THAT(AsSpan(shape), testing::ElementsAreArray(src));
}

TEST(RuntimeShapeTest, TestExtendedShapeSmall) {
  ASSERT_GE(kSmallSize, 2);
  const std::vector<int32_t> dims = IotaVector(kSmallSize - 2);
  const RuntimeShape src(dims.size(), dims.data());
  const RuntimeShape extended = RuntimeShape::ExtendedShape(kSmallSize, src);
  EXPECT_EQ(extended.DimensionsCount(), kSmallSize);
  EXPECT_EQ(extended.Dims(0), 1);
  EXPECT_EQ(extended.Dims(1), 1);
  EXPECT_THAT(absl::Span<const int32_t>(extended.DimsData() + 2, dims.size()),
              ElementsAreArray(dims));
}

TEST(RuntimeShapeTest, TestExtendedShapeBig) {
  ASSERT_GE(kSmallSize, 2);
  const std::vector<int32_t> dims = IotaVector(kBigSize);
  const RuntimeShape src(dims.size(), dims.data());
  const RuntimeShape extended = RuntimeShape::ExtendedShape(kBigSize + 2, src);
  EXPECT_EQ(extended.DimensionsCount(), kBigSize + 2);
  EXPECT_EQ(extended.Dims(0), 1);
  EXPECT_EQ(extended.Dims(1), 1);
  EXPECT_THAT(absl::Span<const int32_t>(extended.DimsData() + 2, dims.size()),
              ElementsAreArray(dims));
}

TEST(RuntimeShapeTest, TestExtendedShapeSmallToBig) {
  const std::vector<int32_t> dims = IotaVector(kSmallSize);
  const RuntimeShape src(dims.size(), dims.data());
  const RuntimeShape extended = RuntimeShape::ExtendedShape(kBigSize, src);
  EXPECT_EQ(extended.DimensionsCount(), kBigSize);
  EXPECT_THAT(
      absl::Span<const int32_t>(extended.DimsData(), kBigSize - kSmallSize),
      Each(1));
  EXPECT_THAT(absl::Span<const int32_t>(
                  extended.DimsData() + kBigSize - kSmallSize, dims.size()),
              ElementsAreArray(dims));
}

TEST_P(RuntimeShapeTest, TestFlatSize) {
  const std::vector<int32_t> src = IotaVector(kSmallSize);
  const RuntimeShape shape(src.size(), src.data());
  int32_t flat_size = 1;
  for (std::vector<int>::const_iterator it = src.begin(); it != src.end(); ++it)
    flat_size *= *it;
  EXPECT_EQ(shape.FlatSize(), flat_size);
}

INSTANTIATE_TEST_SUITE_P(BigSmall, RuntimeShapeTest,
                         testing::Values(kSmallSize, kBigSize),
                         [](const testing::TestParamInfo<int>& info) {
                           return info.param == kSmallSize ? "Small" : "Big";
                         });

}  // namespace
}  // namespace mlir

// LINT.ThenChange(//tensorflow/lite/kernels/internal/runtime_shape_test.cc)
