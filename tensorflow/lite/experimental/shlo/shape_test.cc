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

#include "tensorflow/lite/experimental/shlo/shape.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"

namespace shlo_ref {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;

TEST(ShapeTest, DimensionsAccess) {
  const Shape shape({1, 2, 4, 8});
  EXPECT_THAT(shape.Dimensions(), ElementsAre(1, 2, 4, 8));
}

TEST(ShapeTest, DimensionsMutableAccess) {
  Shape shape({1, 2, 4, 8});

  shape.MutableDimensions()[2] = 42;
  EXPECT_THAT(shape.Dimensions(), ElementsAre(1, 2, 42, 8));
}

TEST(ShapeTest, Axes) {
  Shape shape({1, 2, 4, 8});
  EXPECT_THAT(shape.Axes(), ElementsAre(0, 1, 2, 3));
}

TEST(ShapeTest, Dim) {
  Shape shape({1, 2, 4, 8});
  EXPECT_THAT(shape.Dim(1), Eq(2));
  EXPECT_THAT(shape.Dim(3), Eq(8));
}

TEST(ShapeTest, Dims) {
  Shape shape({1, 2, 4, 8});
  EXPECT_THAT(shape.Dims({1, 3}), ElementsAre(2, 8));
}

TEST(ShapeTest, DimsInvalidAxisIgnored) {
  Shape shape({1, 2, 4, 8});
  EXPECT_THAT(shape.Dims({1, 8}), ElementsAre(2));
}

TEST(ShapeTest, Rank) {
  Shape shape({1, 2, 4, 8});
  EXPECT_THAT(shape.Rank(), Eq(4));
}

TEST(ShapeTest, RankEmpty) {
  Shape shape{};
  EXPECT_THAT(shape.Rank(), Eq(0));
}

TEST(ShapeTest, NumElementsEmpty) {
  Shape shape{};
  EXPECT_THAT(shape.NumElements(), Eq(0));
}
TEST(ShapeTest, NumElements) {
  Shape shape({1, 2, 4, 8});
  EXPECT_THAT(shape.NumElements(), Eq(64));
}

TEST(ShapeTest, Equals) {
  Shape s1({1, 2, 4, 8});
  Shape s2({1, 2, 4, 8});
  EXPECT_TRUE(s1 == s2);
}

TEST(ShapeTest, NotEquals) {
  Shape s1({1, 2, 4, 8});
  Shape s2({1, 4, 2, 8});
  EXPECT_TRUE(s1 != s2);
}

TEST(ShapeTest, ComputeStridesFromShape) {
  const Shape s1({1, 2, 4, 8});
  const Strides expected_strides{64, 32, 8, 1};
  const Strides strides = ComputeStrides(s1);
  EXPECT_EQ(strides, expected_strides);
}

TEST(ShapeTest, ComputeStridesFromSpan) {
  const std::vector<int> s1({1, 2, 4, 5});
  const Strides expected_strides{40, 20, 5, 1};
  const Strides strides = ComputeStrides(absl::Span<const int>(s1));
  EXPECT_EQ(strides, expected_strides);
}

}  // namespace
}  // namespace shlo_ref
