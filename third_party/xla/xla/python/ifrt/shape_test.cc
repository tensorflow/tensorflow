/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/shape.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <sstream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "xla/python/ifrt/shape.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::tsl::testing::StatusIs;

TEST(ShapeTest, LargeDim) {
  Shape shape({std::numeric_limits<int64_t>::max()});
  EXPECT_THAT(shape.dims(), ElementsAre(std::numeric_limits<int64_t>::max()));
}

TEST(ShapeTest, ManyDims) {
  const int kNumDims = 65536;  // Arbitrarily large number.
  std::vector<int64_t> dims(kNumDims);
  std::iota(dims.begin(), dims.end(), 0);
  Shape shape(dims);
  EXPECT_THAT(shape.dims(), ElementsAreArray(dims));
}

TEST(ShapeTest, ScalarNumElements) {
  Shape shape({});
  EXPECT_EQ(shape.num_elements(), 1);
}

TEST(ShapeTest, ZeroDimNumElements) {
  {
    Shape shape({0});
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    Shape shape({1, 0});
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    Shape shape({0, 1});
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    Shape shape({0, 0});
    EXPECT_EQ(shape.num_elements(), 0);
  }
}

TEST(ShapeTest, NonZeroDimsNumElements) {
  {
    Shape shape({2});
    EXPECT_EQ(shape.num_elements(), 2);
  }
  {
    Shape shape({2, 3});
    EXPECT_EQ(shape.num_elements(), 6);
  }
}

TEST(ShapeTest, ToFromProto) {
  {
    Shape shape({});
    ShapeProto proto = shape.ToProto();
    TF_ASSERT_OK_AND_ASSIGN(Shape shape_copy, shape.FromProto(proto));
    EXPECT_EQ(shape_copy, shape);
  }
  {
    Shape shape({1, 2});
    ShapeProto proto = shape.ToProto();
    TF_ASSERT_OK_AND_ASSIGN(Shape shape_copy, shape.FromProto(proto));
    EXPECT_EQ(shape_copy, shape);
  }
}

TEST(BoundedDynamicShapeTagDeathTest, NoDynamicDim) {
  EXPECT_DEATH(BoundedDynamicShapeTag tag({false, false}),
               "At least one dimension needs to be dynamically sized");
}

TEST(BoundedDynamicShapeTagTest, ToFromProto) {
  BoundedDynamicShapeTag tag({true, false});
  BoundedDynamicShapeTagProto proto = tag.ToProto();
  TF_ASSERT_OK_AND_ASSIGN(BoundedDynamicShapeTag tag_copy,
                          tag.FromProto(proto));
  EXPECT_EQ(tag_copy, tag);
}

TEST(DynamicShapeTest, SizeMismatch) {
  Shape shape({1, 2, 3});
  BoundedDynamicShapeTag tag({true, true});
  EXPECT_THAT(DynamicShape::Create(shape, tag),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must have the same number of dimensions")));
}

TEST(DynamicShapeTest, Equality) {
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shape1,
      DynamicShape::Create(Shape({2, 4}),
                           BoundedDynamicShapeTag({true, false})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shape2,
      DynamicShape::Create(Shape({3, 4}),
                           BoundedDynamicShapeTag({true, false})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shape3,
      DynamicShape::Create(Shape({2, 4}),
                           BoundedDynamicShapeTag({true, true})));
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shape4,
      DynamicShape::Create(Shape({2, 4, 3}),
                           BoundedDynamicShapeTag({true, false, true})));
  EXPECT_EQ(shape1, shape1);
  EXPECT_NE(shape1, shape2);
  EXPECT_NE(shape1, shape3);
  EXPECT_NE(shape1, shape4);
}

TEST(DynamicShapeTest, IsDynamicDim) {
  Shape shape({1, 2, 3});
  BoundedDynamicShapeTag tag({true, false, true});
  TF_ASSERT_OK_AND_ASSIGN(DynamicShape dynamic_shape,
                          DynamicShape::Create(shape, tag));
  EXPECT_TRUE(dynamic_shape.IsDynamicDim(0));
  EXPECT_FALSE(dynamic_shape.IsDynamicDim(1));
  EXPECT_TRUE(dynamic_shape.IsDynamicDim(2));
}

TEST(DynamicShapeTest, GetPaddedShape) {
  Shape shape({1, 2, 3});
  BoundedDynamicShapeTag tag({true, true, true});
  TF_ASSERT_OK_AND_ASSIGN(DynamicShape dynamic_shape,
                          DynamicShape::Create(shape, tag));
  TF_ASSERT_OK_AND_ASSIGN(Shape padded_shape, dynamic_shape.GetPaddedShape());
  EXPECT_EQ(padded_shape, shape);
}

TEST(DynamicShapeTest, ToFromProto) {
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shape,
      DynamicShape::Create(Shape({2, 4}),
                           BoundedDynamicShapeTag({true, false})));
  DynamicShapeProto proto = shape.ToProto();
  TF_ASSERT_OK_AND_ASSIGN(DynamicShape shape_copy, shape.FromProto(proto));
  EXPECT_EQ(shape_copy, shape);
}

TEST(DynamicShapeTest, ToString) {
  {
    TF_ASSERT_OK_AND_ASSIGN(
        DynamicShape shape,
        DynamicShape::Create(Shape({2, 4}),
                             BoundedDynamicShapeTag({true, true})));
    std::ostringstream output;
    output << shape;
    EXPECT_EQ(output.str(), "[<=2,<=4]");
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(
        DynamicShape shape,
        DynamicShape::Create(Shape({2, 4}),
                             BoundedDynamicShapeTag({false, true})));
    std::ostringstream output;
    output << shape;
    EXPECT_EQ(output.str(), "[2,<=4]");
  }
}

TEST(ShapeTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      Shape({}),
      Shape({1}),
      Shape({2}),
      Shape({1, 2}),
      Shape({1, 3}),
      Shape({2, 1}),
      Shape({1, 2, 3}),
      Shape({1, 2, 4}),
  }));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
