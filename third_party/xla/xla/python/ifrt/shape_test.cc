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
#include <sstream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.pb.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

TEST(ShapeTest, LargeDim) {
  Shape shape({Shape::kMaxNumElements});
  EXPECT_THAT(shape.dims(), ElementsAre(Shape::kMaxNumElements));
  EXPECT_EQ(shape.num_elements(), Shape::kMaxNumElements);
}

TEST(ShapeTest, ManyDims) {
  const int kNumDims = 65536;  // Arbitrarily large number.
  std::vector<int64_t> dims(kNumDims, 1);
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
  {
    Shape shape({0, Shape::kMaxNumElements});
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    Shape shape({Shape::kMaxNumElements, 0});
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

TEST(ShapeTest, CreateValidDimensions) {
  {
    ASSERT_OK_AND_ASSIGN(Shape shape, Shape::Create({}));
    EXPECT_THAT(shape.dims(), ElementsAre());
    EXPECT_EQ(shape.num_elements(), 1);
  }
  {
    ASSERT_OK_AND_ASSIGN(Shape shape, Shape::Create({2, 3}));
    EXPECT_THAT(shape.dims(), ElementsAre(2, 3));
    EXPECT_EQ(shape.num_elements(), 6);
  }
  {
    ASSERT_OK_AND_ASSIGN(Shape shape, Shape::Create({Shape::kMaxNumElements}));
    EXPECT_THAT(shape.dims(), ElementsAre(Shape::kMaxNumElements));
    EXPECT_EQ(shape.num_elements(), Shape::kMaxNumElements);
  }
  {
    ASSERT_OK_AND_ASSIGN(Shape shape,
                         Shape::Create({0, Shape::kMaxNumElements}));
    EXPECT_EQ(shape.num_elements(), 0);
  }
  {
    ASSERT_OK_AND_ASSIGN(Shape shape,
                         Shape::Create({Shape::kMaxNumElements, 0}));
    EXPECT_EQ(shape.num_elements(), 0);
  }
}

TEST(ShapeTest, CreateRejectsInvalidDimensions) {
  constexpr int64_t kOverflowSide = int64_t{1} << 32;
  EXPECT_THAT(
      Shape::Create({Shape::kMaxNumElements + 1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({0, Shape::kMaxNumElements + 1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({Shape::kMaxNumElements + 1, 0}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({Shape::kMaxNumElements / 2 + 1, 2}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({(1LL << 60) + 1}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({kOverflowSide, kOverflowSide}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({0, kOverflowSide, kOverflowSide}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({kOverflowSide, kOverflowSide, 0}),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr("is too large")));
  EXPECT_THAT(
      Shape::Create({4, -1}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Shape expects non-negative dimension sizes")));
  EXPECT_THAT(
      Shape::Create({0, -1}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Shape expects non-negative dimension sizes")));
  EXPECT_THAT(
      Shape::Create({std::numeric_limits<int64_t>::min(), -1}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Shape expects non-negative dimension sizes")));
}

TEST(ShapeDeathTest, ConstructorFailsOnOverflowingElementCount) {
  constexpr int64_t kOverflowSide = int64_t{1} << 32;
  EXPECT_DEATH((void)Shape({Shape::kMaxNumElements + 1}), "is too large");
  EXPECT_DEATH((void)Shape({0, Shape::kMaxNumElements + 1}), "is too large");
  EXPECT_DEATH((void)Shape({Shape::kMaxNumElements / 2 + 1, 2}),
               "is too large");
  EXPECT_DEATH((void)Shape({(1LL << 60) + 1}), "is too large");
  EXPECT_DEATH((void)Shape({kOverflowSide, kOverflowSide}), "is too large");
  EXPECT_DEATH((void)Shape({0, kOverflowSide, kOverflowSide}), "is too large");
  EXPECT_DEATH((void)Shape({kOverflowSide, kOverflowSide, 0}), "is too large");
  EXPECT_DEATH((void)Shape({kOverflowSide, kOverflowSide, 5}), "is too large");
}

TEST(ShapeDeathTest, ConstructorFailsOnNegativeDimensions) {
  EXPECT_DEATH((void)Shape({4, -1}),
               "Shape expects non-negative dimension sizes");
  EXPECT_DEATH((void)Shape({0, -1}),
               "Shape expects non-negative dimension sizes");
  EXPECT_DEATH((void)Shape({-2, -3}),
               "Shape expects non-negative dimension sizes");
  EXPECT_DEATH((void)Shape({std::numeric_limits<int64_t>::min(), -1}),
               "Shape expects non-negative dimension sizes");
}

class ShapeSerDesTest : public testing::TestWithParam<SerDesVersion> {
 public:
  ShapeSerDesTest() : version_(GetParam()) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

TEST_P(ShapeSerDesTest, ToFromProto) {
  {
    Shape shape({});
    ShapeProto proto = shape.ToProto(version());
    TF_ASSERT_OK_AND_ASSIGN(Shape shape_copy, shape.FromProto(proto));
    EXPECT_EQ(shape_copy, shape);
  }
  {
    Shape shape({1, 2});
    ShapeProto proto = shape.ToProto(version());
    TF_ASSERT_OK_AND_ASSIGN(Shape shape_copy, shape.FromProto(proto));
    EXPECT_EQ(shape_copy, shape);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, ShapeSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
    [](const testing::TestParamInfo<SerDesVersion>& info) {
      return absl::StrCat(info.param.version_number().value());
    });

TEST(ShapeTest, FromProtoRejectsInvalidDimensions) {
  {
    // Individually plausible dimensions whose product overflows int64_t.
    ShapeProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.add_dims(3037000500LL);
    proto.add_dims(3037000500LL);
    proto.add_dims(4LL);

    EXPECT_THAT(Shape::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("is too large")));
  }
  {
    // Product fits in int64_t, but exceeds kMaxNumElements (so 128-bit bit-size
    // calculation would overflow int64_t).
    ShapeProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.add_dims(Shape::kMaxNumElements + 1);

    EXPECT_THAT(Shape::FromProto(proto),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("is too large")));
  }
  {
    ShapeProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.add_dims(4);
    proto.add_dims(-1);

    EXPECT_THAT(
        Shape::FromProto(proto),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("Shape expects non-negative dimension sizes")));
  }
}

TEST(ShapeTest, FromProtoAcceptsNonOverflowingDimensions) {
  ShapeProto proto;
  proto.set_version_number(SerDesVersionNumber(0).value());
  proto.add_dims(100);
  proto.add_dims(200);

  ASSERT_OK_AND_ASSIGN(Shape shape, Shape::FromProto(proto));
  EXPECT_EQ(shape, Shape({100, 200}));
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

TEST(BoundedDynamicShapeTagTest, Create) {
  ASSERT_OK_AND_ASSIGN(BoundedDynamicShapeTag tag,
                       BoundedDynamicShapeTag::Create({false, true}));
  EXPECT_THAT(tag.DynamicDims(), ElementsAre(false, true));

  EXPECT_THAT(
      BoundedDynamicShapeTag::Create({}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("At least one dimension needs to be dynamically")));
  EXPECT_THAT(
      BoundedDynamicShapeTag::Create({false, false}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("At least one dimension needs to be dynamically")));
}

TEST(BoundedDynamicShapeTagDeathTest, NoDynamicDim) {
  EXPECT_DEATH((void)BoundedDynamicShapeTag({}),
               "At least one dimension needs to be dynamically sized");
  EXPECT_DEATH((void)BoundedDynamicShapeTag({false, false}),
               "At least one dimension needs to be dynamically sized");
}

class BoundedDynamicShapeTagSerDesTest : public ShapeSerDesTest {};

TEST_P(BoundedDynamicShapeTagSerDesTest, ToFromProto) {
  BoundedDynamicShapeTag tag({true, false});
  BoundedDynamicShapeTagProto proto = tag.ToProto(version());
  ASSERT_OK_AND_ASSIGN(BoundedDynamicShapeTag tag_copy,
                       BoundedDynamicShapeTag::FromProto(proto));
  EXPECT_EQ(tag_copy, tag);
}

TEST(BoundedDynamicShapeTagTest, FromProtoRejectsStaticOrEmptyDims) {
  {
    BoundedDynamicShapeTagProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    EXPECT_THAT(
        BoundedDynamicShapeTag::FromProto(proto),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("At least one dimension needs to be dynamically")));
  }
  {
    BoundedDynamicShapeTagProto proto;
    proto.set_version_number(SerDesVersionNumber(0).value());
    proto.add_is_dynamic_dims(false);
    proto.add_is_dynamic_dims(false);
    EXPECT_THAT(
        BoundedDynamicShapeTag::FromProto(proto),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 HasSubstr("At least one dimension needs to be dynamically")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, BoundedDynamicShapeTagSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
    [](const testing::TestParamInfo<SerDesVersion>& info) {
      return absl::StrCat(info.param.version_number().value());
    });

TEST(DynamicShapeTest, SizeMismatch) {
  Shape shape({1, 2, 3});
  BoundedDynamicShapeTag tag({true, true});
  EXPECT_THAT(DynamicShape::Create(shape, tag),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
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

class DynamicShapeSerDesTest : public ShapeSerDesTest {};

TEST_P(DynamicShapeSerDesTest, ToFromProto) {
  TF_ASSERT_OK_AND_ASSIGN(
      DynamicShape shape,
      DynamicShape::Create(Shape({2, 4}),
                           BoundedDynamicShapeTag({true, false})));
  DynamicShapeProto proto = shape.ToProto(version());
  TF_ASSERT_OK_AND_ASSIGN(DynamicShape shape_copy, shape.FromProto(proto));
  EXPECT_EQ(shape_copy, shape);
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion, DynamicShapeSerDesTest,
    testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
    [](const testing::TestParamInfo<SerDesVersion>& info) {
      return absl::StrCat(info.param.version_number().value());
    });

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

TEST(DynamicShapeTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      DynamicShape::Create(Shape({1}), BoundedDynamicShapeTag({true})).value(),
      DynamicShape::Create(Shape({1, 2}), BoundedDynamicShapeTag({true, false}))
          .value(),
  }));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
