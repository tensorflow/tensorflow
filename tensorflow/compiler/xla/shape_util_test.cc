/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/shape_util.h"

#include <numeric>
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(ShapeUtilTest, ShapeIndexViewTest) {
  ShapeIndex index = {1, 2, 3, 4};
  ShapeIndexView index_view(index, 1);
  EXPECT_EQ(3, index_view.size());
  EXPECT_EQ(ShapeIndexView({2, 3, 4}), index_view);
  EXPECT_EQ(ShapeIndexView({3, 4}), index_view.ConsumeFront());
  EXPECT_EQ(ShapeIndexView({2, 3}), index_view.ConsumeBack());
}

TEST(ShapeUtilTest, GetDimensionHelperCanNegativeIndex) {
  Shape matrix = ShapeUtil::MakeShape(F32, {2, 3});
  EXPECT_EQ(3, ShapeUtil::GetDimension(matrix, -1));
  EXPECT_EQ(2, ShapeUtil::GetDimension(matrix, -2));
}

TEST(ShapeUtilTest, GetDimensionHelperExampleInDocumentationTest) {
  auto shape = ShapeUtil::MakeShape(F32, {1, 2, 3, 4});
  ASSERT_EQ(4, ShapeUtil::GetDimension(shape, -1));
}

TEST(ShapeUtilTest, NegativeIndexOobFails) {
  Shape matrix = ShapeUtil::MakeShape(F32, {2, 3});
  ASSERT_DEATH(ShapeUtil::GetDimension(matrix, -3), "dimension_number >= 0");
}

TEST(ShapeUtilTest, Rank1DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3});
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank2DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2});
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank3DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2, 7});
  ASSERT_EQ(7, shape.dimensions(2));
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, Rank4DimensionIndexing) {
  Shape shape = ShapeUtil::MakeShape(F32, {3, 2, 7, 8});
  ASSERT_EQ(8, shape.dimensions(3));
  ASSERT_EQ(7, shape.dimensions(2));
  ASSERT_EQ(2, shape.dimensions(1));
  ASSERT_EQ(3, shape.dimensions(0));
}

TEST(ShapeUtilTest, ParseShapeStringR2F32) {
  string shape_string = "f32[123,456]";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual,
                          ShapeUtil::ParseShapeString(shape_string));
  Shape expected = ShapeUtil::MakeShape(F32, {123, 456});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseShapeStringTupleOfArrays) {
  string shape_string = "(f32[1572864],s8[5120,1024])";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual,
                          ShapeUtil::ParseShapeString(shape_string));
  Shape expected =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1572864}),
                                 ShapeUtil::MakeShape(S8, {5120, 1024})});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseShapeStringNestedTuple) {
  string shape_string = "(f32[1],(f32[2], token[]), opaque[], f32[3])";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual,
                          ShapeUtil::ParseShapeString(shape_string));
  Shape expected = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShape(F32, {1}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeShape(F32, {2}), ShapeUtil::MakeTokenShape()}),
      ShapeUtil::MakeOpaqueShape(),
      ShapeUtil::MakeShape(F32, {3}),
  });
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseShapeStringWithLayout) {
  string shape_string = "f32[123,456]{0,1}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual,
                          ShapeUtil::ParseShapeString(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithLayout(F32, {123, 456}, {0, 1});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseShapeStringWithExplicitDenseLayout) {
  string shape_string = "f32[123,456]dense{0,1}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual,
                          ShapeUtil::ParseShapeString(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithLayout(F32, {123, 456}, {0, 1});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseShapeStringWithSparseLayout) {
  string shape_string = "f32[123,456]sparse{10}";
  TF_ASSERT_OK_AND_ASSIGN(Shape actual,
                          ShapeUtil::ParseShapeString(shape_string));
  Shape expected = ShapeUtil::MakeShapeWithSparseLayout(F32, {123, 456}, 10);
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual: " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseOpaqueType) {
  TF_ASSERT_OK_AND_ASSIGN(Shape actual,
                          ShapeUtil::ParseShapeString("opaque[]"));
  Shape expected = ShapeUtil::MakeOpaqueShape();
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseTokenType) {
  TF_ASSERT_OK_AND_ASSIGN(Shape actual, ShapeUtil::ParseShapeString("token[]"));
  Shape expected = ShapeUtil::MakeTokenShape();
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, ParseInvalidShapeString) {
  string shape_strings[] = {
      "f32[123,456]foobar{0,1}", "f32[123,456]sparse{0,1}", "f32[123,456]{foo}",
      "f32[123,456]dense{foo}",  "f32[123,456]sparse{foo}",
  };
  for (const string& shape_string : shape_strings) {
    StatusOr<Shape> result = ShapeUtil::ParseShapeString(shape_string);
    ASSERT_FALSE(result.ok()) << "shape: " << shape_string;
  }
}

TEST(ShapeUtilTest, CompatibleIdenticalShapes) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  ASSERT_TRUE(ShapeUtil::Compatible(shape1, shape2));
}

TEST(ShapeUtilTest, TokenCompatibility) {
  EXPECT_TRUE(ShapeUtil::Compatible(ShapeUtil::MakeTokenShape(),
                                    ShapeUtil::MakeTokenShape()));
  EXPECT_FALSE(ShapeUtil::Compatible(ShapeUtil::MakeTokenShape(),
                                     ShapeUtil::MakeShape(F32, {})));
  EXPECT_FALSE(ShapeUtil::Compatible(ShapeUtil::MakeShape(F32, {}),
                                     ShapeUtil::MakeTokenShape()));
  EXPECT_TRUE(ShapeUtil::Compatible(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape()}),
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape()})));
}

TEST(ShapeUtilTest, TokensEqualShapes) {
  EXPECT_TRUE(ShapeUtil::Equal(ShapeUtil::MakeTokenShape(),
                               ShapeUtil::MakeTokenShape()));
  EXPECT_FALSE(ShapeUtil::Equal(ShapeUtil::MakeTokenShape(),
                                ShapeUtil::MakeShape(F32, {})));
  EXPECT_FALSE(ShapeUtil::Equal(ShapeUtil::MakeShape(F32, {}),
                                ShapeUtil::MakeTokenShape()));
  EXPECT_TRUE(ShapeUtil::Equal(
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithLayout(S32, {3, 4}, {0, 1})}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithLayout(S32, {3, 4}, {0, 1})})));
  EXPECT_FALSE(ShapeUtil::Equal(
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithLayout(S32, {3, 4}, {0, 1})}),
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::MakeTokenShape(),
           ShapeUtil::MakeShapeWithLayout(S32, {3, 4}, {1, 0})})));
}

TEST(ShapeUtilTest, CompatibleNotIdenticalShapes) {
  Shape shape_1 = ShapeUtil::MakeShape(F32, {3, 2});
  auto layout_1 = shape_1.mutable_layout();
  layout_1->clear_minor_to_major();
  layout_1->add_minor_to_major(0);
  layout_1->add_minor_to_major(1);

  Shape shape_2 = ShapeUtil::MakeShape(F32, {3, 2});
  auto layout_2 = shape_2.mutable_layout();
  layout_2->clear_minor_to_major();
  layout_2->add_minor_to_major(1);
  layout_2->add_minor_to_major(0);

  EXPECT_FALSE(ShapeUtil::Equal(shape_1, shape_2));
  EXPECT_TRUE(ShapeUtil::Compatible(shape_1, shape_2));
}

TEST(ShapeUtilTest, CompatibleIgnoringFpPrecision) {
  Shape shape1 = ShapeUtil::MakeShape(BF16, {3, 2});
  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  ASSERT_TRUE(ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2));
}

TEST(ShapeUtilTest, IncompatibleIgnoringFpPrecision) {
  Shape shape1 = ShapeUtil::MakeShape(BF16, {3, 2});
  Shape shape2 = ShapeUtil::MakeShape(F32, {2, 2});
  ASSERT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2));
}

TEST(ShapeUtilTest, IncompatibleDifferentElementShapes) {
  Shape shape_1 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape_2 = ShapeUtil::MakeShape(PRED, {3, 2});
  EXPECT_FALSE(ShapeUtil::Compatible(shape_1, shape_2));
}

TEST(ShapeUtilTest, EqualIgnoringFpPrecision) {
  EXPECT_TRUE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithLayout(F16, {4, 3}, {0, 1})));
}

TEST(ShapeUtilTest, UnequalIgnoringFpPrecision) {
  EXPECT_FALSE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithLayout(F16, {3, 4}, {0, 1})));
  EXPECT_FALSE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithLayout(F32, {3, 4}, {0, 1}),
      ShapeUtil::MakeShapeWithLayout(F16, {3, 4}, {1, 0})));
  EXPECT_FALSE(ShapeUtil::EqualIgnoringFpPrecision(
      ShapeUtil::MakeShapeWithLayout(F32, {4, 3}, {0, 1}),
      ShapeUtil::MakeShapeWithLayout(PRED, {4, 3}, {0, 1})));
}

TEST(ShapeUtilTest, CompatibleTuples) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  EXPECT_TRUE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, CompatibleTuplesIgnoringFpPrecision) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(BF16, {3, 2}), ShapeUtil::MakeShape(F32, {4, 5})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F64, {3, 2}), ShapeUtil::MakeShape(BF16, {4, 5})});
  EXPECT_TRUE(ShapeUtil::CompatibleIgnoringFpPrecision(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithSwappedElements) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesIgnoringFpPrecision) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(BF16, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(BF16, {4, 5})});
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithDifferentPrimitiveType) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(S32, {3, 2})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
  EXPECT_TRUE(ShapeUtil::CompatibleIgnoringElementType(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithDifferentDimensions) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {4, 2})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleScalarVsTuple) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {});
  Shape shape2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(U32, {})});
  EXPECT_FALSE(ShapeUtil::Compatible(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::Compatible(shape2, shape1));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringElementType(shape2, shape1));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape1, shape2));
  EXPECT_FALSE(ShapeUtil::CompatibleIgnoringFpPrecision(shape2, shape1));
}

TEST(ShapeUtilTest, CompareShapesWithPaddedDimensionsMismatch) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {20, 30});
  shape1.mutable_layout()->add_padded_dimensions(10);

  Shape shape2 = ShapeUtil::MakeShape(F32, {20, 30});
  shape2.mutable_layout()->add_padded_dimensions(11);

  EXPECT_FALSE(ShapeUtil::Equal(shape1, shape2));
}

TEST(ShapeUtilTest, CompareShapesWithPaddingValueMismatch) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {20, 30});
  shape1.mutable_layout()->set_padding_value(ZERO_PAD);

  Shape shape2 = ShapeUtil::MakeShape(F32, {20, 30});
  shape2.mutable_layout()->set_padding_value(LOWEST_PAD);

  EXPECT_FALSE(ShapeUtil::Equal(shape1, shape2));
}

TEST(ShapeUtilTest, ScalarDefaultLayoutEqualsScalarEmptyMin2Maj) {
  Shape scalar_default_layout = ShapeUtil::MakeShape(F32, {});
  ASSERT_TRUE(scalar_default_layout.has_layout())
      << ShapeUtil::HumanStringWithLayout(scalar_default_layout);

  const Shape scalar_empty_min2maj =
      ShapeUtil::MakeShapeWithLayout(F32, {}, {});
  ASSERT_TRUE(scalar_empty_min2maj.has_layout())
      << ShapeUtil::HumanStringWithLayout(scalar_empty_min2maj);

  EXPECT_TRUE(ShapeUtil::Equal(scalar_default_layout, scalar_empty_min2maj));
}

TEST(ShapeUtilTest, ByteSizeOfWithoutPadding) {
  EXPECT_EQ(4, ShapeUtil::ByteSizeOfPrimitiveType(F32));
  EXPECT_EQ(4, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F32, {})));
  EXPECT_EQ(800, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F32, {10, 20})));

  EXPECT_EQ(8, ShapeUtil::ByteSizeOfPrimitiveType(F64));
  EXPECT_EQ(8, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F64, {})));
  EXPECT_EQ(1600, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F64, {10, 20})));

  EXPECT_EQ(8, ShapeUtil::ByteSizeOfPrimitiveType(C64));
  EXPECT_EQ(8, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(C64, {})));
  EXPECT_EQ(1600, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(C64, {10, 20})));

  EXPECT_EQ(0, ShapeUtil::ByteSizeOfPrimitiveType(TOKEN));
  EXPECT_EQ(0, ShapeUtil::ByteSizeOf(ShapeUtil::MakeTokenShape()));
}

TEST(ShapeUtilTest, ByteSizeOfWithPadding) {
  EXPECT_EQ(4, ShapeUtil::ByteSizeOfPrimitiveType(F32));
  Shape shape = ShapeUtil::MakeShape(F32, {10, 20});
  EXPECT_EQ(800, ShapeUtil::ByteSizeOf(shape));

  shape.mutable_layout()->add_padded_dimensions(15);
  shape.mutable_layout()->add_padded_dimensions(21);
  EXPECT_EQ(15 * 21 * 4, ShapeUtil::ByteSizeOf(shape));
}

TEST(ShapeUtilTest, NilShape) {
  EXPECT_TRUE(ShapeUtil::IsNil(ShapeUtil::MakeNil()));
  EXPECT_FALSE(ShapeUtil::IsNil(ShapeUtil::MakeShape(F32, {1, 2, 3})));
  EXPECT_FALSE(ShapeUtil::IsNil(ShapeUtil::MakeShape(F32, {0, 1})));
  EXPECT_FALSE(ShapeUtil::IsNil(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {})})));
  EXPECT_FALSE(ShapeUtil::IsNil(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {0})})));
}

TEST(ShapeUtilTest, NestedTuple) {
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape({})));
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeTupleShape({})})));
  EXPECT_FALSE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(S32, {})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeTupleShape({})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({}), ShapeUtil::MakeShape(S32, {})})));
  EXPECT_TRUE(ShapeUtil::IsNestedTuple(ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({}), ShapeUtil::MakeTupleShape({})})));
}

TEST(ShapeUtilTest, ElementsIn) {
  EXPECT_EQ(1, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {})));
  EXPECT_EQ(0, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {0})));
  EXPECT_EQ(1, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {1})));
  EXPECT_EQ(1, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {1, 1})));
  EXPECT_EQ(2, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {2})));
  EXPECT_EQ(2, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {2, 1})));
  EXPECT_EQ(15, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {3, 5})));
  EXPECT_EQ(0, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {3, 0, 5})));
  EXPECT_EQ(0, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {0, 3, 0})));
  EXPECT_EQ(15, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {1, 3, 5})));
  EXPECT_EQ(221, ShapeUtil::ElementsIn(ShapeUtil::MakeShape(S32, {13, 17})));
}

TEST(ShapeUtilTest, IsZeroElementArray) {
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {})));
  EXPECT_TRUE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {0})));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {1})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {1, 1})));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {2})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {2, 1})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {3, 5})));
  EXPECT_TRUE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {3, 0, 5})));
  EXPECT_TRUE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {0, 3, 0})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {1, 3, 5})));
  EXPECT_FALSE(
      ShapeUtil::IsZeroElementArray(ShapeUtil::MakeShape(S32, {13, 17})));

  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeNil()));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(ShapeUtil::MakeTupleShape({})));
  EXPECT_FALSE(ShapeUtil::IsZeroElementArray(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(S32, {0, 3, 0})})));
}

TEST(ShapeUtilTest, SameDimensions) {
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {}),
                                        ShapeUtil::MakeShape(S32, {})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {}),
                                        ShapeUtil::MakeShape(F32, {})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                        ShapeUtil::MakeShape(S32, {1})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {0}),
                                        ShapeUtil::MakeShape(S32, {0})));
  EXPECT_TRUE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {2}),
                                        ShapeUtil::MakeShape(S32, {2})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {2})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {0, 0}),
                                         ShapeUtil::MakeShape(F32, {0})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {1, 1})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {}),
                                         ShapeUtil::MakeShape(F32, {1})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {1, 1})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1}),
                                         ShapeUtil::MakeShape(F32, {1, 0})));
  EXPECT_FALSE(ShapeUtil::SameDimensions(ShapeUtil::MakeShape(S32, {1, 1}),
                                         ShapeUtil::MakeShape(F32, {1, 2})));
}

TEST(ShapeUtilTest, GetSubshape) {
  // Test array shape.
  Shape array_shape = ShapeUtil::MakeShape(F32, {42, 42, 123});
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(array_shape, {})));
  EXPECT_TRUE(ShapeUtil::Equal(
      array_shape, *ShapeUtil::GetMutableSubshape(&array_shape, {})));

  // Test tuple shape.
  Shape tuple_shape =
      ShapeUtil::MakeTupleShape({array_shape, array_shape, array_shape});
  EXPECT_TRUE(
      ShapeUtil::Equal(tuple_shape, ShapeUtil::GetSubshape(tuple_shape, {})));
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(tuple_shape, {0})));
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(tuple_shape, {1})));
  EXPECT_TRUE(
      ShapeUtil::Equal(array_shape, ShapeUtil::GetSubshape(tuple_shape, {2})));

  // Test nested tuple shape.
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape(
      {array_shape, ShapeUtil::MakeTupleShape({array_shape, array_shape}),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeTupleShape({array_shape, array_shape}),
            array_shape})});
  EXPECT_TRUE(ShapeUtil::Equal(nested_tuple_shape,
                               ShapeUtil::GetSubshape(nested_tuple_shape, {})));
  EXPECT_TRUE(ShapeUtil::Equal(
      array_shape, ShapeUtil::GetSubshape(nested_tuple_shape, {0})));
  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeTupleShape({array_shape, array_shape}),
                       ShapeUtil::GetSubshape(nested_tuple_shape, {1})));
  EXPECT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeTupleShape({array_shape, array_shape}),
                       ShapeUtil::GetSubshape(nested_tuple_shape, {2, 0})));
}

TEST(ShapeUtilTest, IsLeafIndex) {
  // Test array shape.
  Shape array_shape = ShapeUtil::MakeShape(F32, {42, 42, 123});
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(array_shape, {}));

  // Test tuple shape.
  Shape tuple_shape = ShapeUtil::MakeTupleShape({array_shape, array_shape});
  EXPECT_FALSE(ShapeUtil::IsLeafIndex(tuple_shape, {}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(tuple_shape, {0}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(tuple_shape, {1}));

  // Test nested tuple shape.
  Shape nested_tuple_shape = ShapeUtil::MakeTupleShape(
      {array_shape, ShapeUtil::MakeTupleShape({array_shape, array_shape}),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeTupleShape({array_shape, array_shape}),
            array_shape})});
  EXPECT_FALSE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {0}));
  EXPECT_FALSE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {1}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {1, 0}));
  EXPECT_TRUE(ShapeUtil::IsLeafIndex(nested_tuple_shape, {1, 1}));
}

TEST(ShapeUtilTest, HumanString) {
  Shape opaque = ShapeUtil::MakeOpaqueShape();
  Shape token = ShapeUtil::MakeTokenShape();
  Shape scalar = ShapeUtil::MakeShape(F32, {});
  Shape matrix = ShapeUtil::MakeShape(U32, {1, 2});
  Shape matrix2 = ShapeUtil::MakeShapeWithLayout(S32, {3, 4}, {0, 1});
  Shape tuple = ShapeUtil::MakeTupleShape({opaque, scalar, matrix, matrix2});
  Shape nested_tuple = ShapeUtil::MakeTupleShape({tuple, matrix, token});

  EXPECT_EQ("opaque[]", ShapeUtil::HumanString(opaque));
  EXPECT_EQ("token[]", ShapeUtil::HumanString(token));
  EXPECT_EQ("f32[]", ShapeUtil::HumanString(scalar));
  EXPECT_EQ("u32[1,2]", ShapeUtil::HumanString(matrix));
  EXPECT_EQ("s32[3,4]", ShapeUtil::HumanString(matrix2));
  EXPECT_EQ("(opaque[], f32[], u32[1,2], s32[3,4])",
            ShapeUtil::HumanString(tuple));
  EXPECT_EQ("((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
            ShapeUtil::HumanString(nested_tuple));

  EXPECT_EQ("opaque[]", ShapeUtil::HumanStringWithLayout(opaque));
  EXPECT_EQ("f32[]", ShapeUtil::HumanStringWithLayout(scalar));
  EXPECT_EQ("u32[1,2]{1,0}", ShapeUtil::HumanStringWithLayout(matrix));
  EXPECT_EQ("s32[3,4]{0,1}", ShapeUtil::HumanStringWithLayout(matrix2));
  EXPECT_EQ("(opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1})",
            ShapeUtil::HumanStringWithLayout(tuple));
  EXPECT_EQ(
      "((opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1}), u32[1,2]{1,0}, "
      "token[])",
      ShapeUtil::HumanStringWithLayout(nested_tuple));

  ProgramShape prog = ShapeUtil::MakeProgramShape(
      {opaque, scalar, matrix, matrix2, tuple, nested_tuple}, nested_tuple);
  EXPECT_EQ(
      "((unknown): opaque[], "
      "(unknown): f32[], "
      "(unknown): u32[1,2], "
      "(unknown): s32[3,4], "
      "(unknown): (opaque[], f32[], u32[1,2], s32[3,4]), "
      "(unknown): ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      ShapeUtil::HumanString(prog));

  prog.add_parameter_names("arg0");
  prog.add_parameter_names("scalar");
  prog.add_parameter_names("matrix");
  prog.add_parameter_names("matrix2");
  prog.add_parameter_names("tuple");
  prog.add_parameter_names("nested_tuple");
  EXPECT_EQ(
      "(arg0: opaque[], "
      "scalar: f32[], "
      "matrix: u32[1,2], "
      "matrix2: s32[3,4], "
      "tuple: (opaque[], f32[], u32[1,2], s32[3,4]), "
      "nested_tuple: ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], "
      "token[])) "
      "-> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2], token[])",
      ShapeUtil::HumanString(prog));
}

TEST(ShapeUtilTest, ForEachSubshapeArray) {
  const Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  int calls = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&calls, &shape](const Shape& subshape, const ShapeIndex& index) {
        EXPECT_EQ(&shape, &subshape);
        EXPECT_TRUE(index.empty());
        ++calls;
      });
  EXPECT_EQ(1, calls);
}

TEST(ShapeUtilTest, ForEachSubshapeNestedTuple) {
  const Shape shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {101}),
                                  ShapeUtil::MakeShape(PRED, {33})})});
  int calls = 0;
  ShapeUtil::ForEachSubshape(
      shape, [&calls, &shape](const Shape& subshape, const ShapeIndex& index) {
        EXPECT_TRUE(
            ShapeUtil::Equal(subshape, ShapeUtil::GetSubshape(shape, index)));
        if (calls == 0) {
          // Visitation should go from outside in.
          EXPECT_TRUE(index.empty());
        } else if (calls == 4) {
          // Last visitation should be to the array with 33 elements.
          EXPECT_EQ(33, ShapeUtil::ElementsIn(subshape));
        }
        ++calls;
      });
  EXPECT_EQ(5, calls);
}

TEST(ShapeUtilTest, ForEachMutableSubshapeNestedTuple) {
  Shape shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {101}),
                                  ShapeUtil::MakeShape(PRED, {33})})});
  int calls = 0;
  ShapeUtil::ForEachMutableSubshape(
      &shape, [&calls, &shape](const Shape* subshape, const ShapeIndex& index) {
        // Pointer values should be equal
        EXPECT_EQ(subshape, ShapeUtil::GetMutableSubshape(&shape, index));
        if (calls == 0) {
          // Visitation should go from outside in.
          EXPECT_TRUE(index.empty());
        } else if (calls == 4) {
          // Last visitation should be to the array with 33 elements.
          EXPECT_EQ(33, ShapeUtil::ElementsIn(*subshape));
        }
        ++calls;
      });
  EXPECT_EQ(5, calls);
}

TEST(ShapeUtilTest, InsertedOrDeleted1SizedDimensions) {
  Shape shape0 = ShapeUtil::MakeShape(S32, {9, 1, 4});
  Shape shape1 = ShapeUtil::MakeShape(S32, {1, 9, 4, 1});
  Shape shape2 = ShapeUtil::MakeShape(S32, {3, 1, 12});
  EXPECT_TRUE(std::get<0>(
      ShapeUtil::InsertedOrDeleted1SizedDimensions(shape0, shape1)));
  EXPECT_FALSE(std::get<0>(
      ShapeUtil::InsertedOrDeleted1SizedDimensions(shape0, shape2)));
}

TEST(ShapeUtilTest, ShapeIs) {
  EXPECT_FALSE(ShapeUtil::ShapeIs(ShapeUtil::MakeShape(PRED, {2}), PRED, {}));
}

TEST(ShapeUtilTest, ForEachIndex) {
  struct ShapeDimensionAndNumberInvocations {
    std::vector<int64> dimensions;
    int invocations;
  } test_data[] = {
      {{}, 1},     {{0}, 0},      {{16}, 16},          {{3, 0}, 0},
      {{0, 2}, 0}, {{4, 16}, 64}, {{6, 11, 17}, 1122}, {{6, 11, 5, 17}, 5610},
  };

  for (const auto& data : test_data) {
    Shape shape = ShapeUtil::MakeShape(F32, data.dimensions);
    // Increments at every invocation.
    int invocations = 0;
    auto increment_func =
        [&invocations](tensorflow::gtl::ArraySlice<int64> indexes) {
          invocations++;
          return true;
        };

    std::vector<int64> zero_base(data.dimensions.size(), 0);
    std::vector<int64> step(data.dimensions.size(), 1);

    ShapeUtil::ForEachIndex(shape, zero_base, data.dimensions, step,
                            increment_func);

    EXPECT_EQ(invocations, data.invocations);
  }
}

TEST(ShapeUtilTest, ForEachIndexWithStatus) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});
  // Increments at every invocation.
  int invocations = 0;
  auto increment_func =
      [&invocations](
          tensorflow::gtl::ArraySlice<int64> indexes) -> StatusOr<bool> {
    if (++invocations == 5) {
      return Unimplemented("Cannot increment beyond 5.");
    }
    return true;
  };

  Status error_status = ShapeUtil::ForEachIndexWithStatus(
      shape, /*base=*/{0, 0}, /*count=*/{10, 10}, /*incr=*/{0, 1},
      increment_func);

  EXPECT_FALSE(error_status.ok());
  EXPECT_THAT(error_status.error_message(),
              ::testing::HasSubstr("Cannot increment beyond 5."));
  EXPECT_EQ(invocations, 5);
}

TEST(ShapeUtilTest, ForEachIndexParallel) {
  Shape shape = ShapeUtil::MakeShape(F32, {10, 10});
  int64 output[10][10];
  int init = 5;
  auto set_func = [&](tensorflow::gtl::ArraySlice<int64> indexes) {
    output[indexes[0]][indexes[1]] = init + indexes[0] + indexes[1];
  };

  ShapeUtil::ForEachIndexParallel(shape, /*base=*/{0, 0}, /*count=*/{10, 10},
                                  /*incr=*/{1, 1}, set_func);

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      EXPECT_EQ(output[i][j], init + i + j);
    }
  }
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1x1_to_1x1x1) {
  // All output dimensions should be unmodified. One of the input dimensions is
  // modified because the input rank is larger by one.
  EXPECT_THAT(ShapeUtil::DimensionsUnmodifiedByReshape(
                  ShapeUtil::MakeShape(S32, {1, 1, 1, 1}),
                  ShapeUtil::MakeShape(S32, {1, 1, 1})),
              ElementsAre(std::make_pair(0, 0), std::make_pair(1, 1),
                          std::make_pair(2, 2)));
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1_to_1x1x1x1) {
  // All input dimensions should be unmodified. One of the output dimensions is
  // modified because the output rank is larger by one.
  EXPECT_THAT(ShapeUtil::DimensionsUnmodifiedByReshape(
                  ShapeUtil::MakeShape(S32, {1, 1, 1}),
                  ShapeUtil::MakeShape(S32, {1, 1, 1, 1})),
              ElementsAre(std::make_pair(0, 0), std::make_pair(1, 1),
                          std::make_pair(2, 2)));
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_4x1x3x5x6x7_to_2x6x1x5x1x42) {
  // The only matching dimension is the one with size 5.
  // 4, 1, 3, 5, 6, 7
  //          |
  // 2, 6, 1, 5, 1, 42
  EXPECT_THAT(ShapeUtil::DimensionsUnmodifiedByReshape(
                  ShapeUtil::MakeShape(S32, {4, 1, 3, 5, 6, 7}),
                  ShapeUtil::MakeShape(S32, {2, 6, 1, 5, 1, 42})),
              ElementsAre(std::make_pair(3, 3)));
}

TEST(ShapeUtilTest, ReshapeIsBitcast_3x4_6x2) {
  for (bool input_is_row_major : {true, false}) {
    for (bool output_is_row_major : {true, false}) {
      Layout input_layout = input_is_row_major ? LayoutUtil::MakeLayout({1, 0})
                                               : LayoutUtil::MakeLayout({0, 1});
      Layout output_layout = output_is_row_major
                                 ? LayoutUtil::MakeLayout({1, 0})
                                 : LayoutUtil::MakeLayout({0, 1});
      // Suppose the input is logically (i.e. ignoring its layout)
      //   0  1  2  3
      //   4  5  6  7
      //   8  9  10 11
      //
      // The reshape transforms the input to logically
      //   0  1
      //   2  3
      //   4  5
      //   6  7
      //   8  9
      //   10 11
      //
      // The input and the output have the same underlying data only if they
      // are both row-major.
      EXPECT_EQ(
          ShapeUtil::ReshapeIsBitcast(
              ShapeUtil::MakeShapeWithLayout(
                  F32, {3, 4}, AsInt64Slice(input_layout.minor_to_major())),
              ShapeUtil::MakeShapeWithLayout(
                  F32, {6, 2}, AsInt64Slice(output_layout.minor_to_major()))),
          input_is_row_major && output_is_row_major);
    }
  }
}

TEST(ShapeUtilTest, ReshapeIsBitcast_3x2x2_6x2_Dim1IsMostMinor) {
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithLayout(F32, {3, 2, 2}, {1, 0, 2}),
      ShapeUtil::MakeShapeWithLayout(F32, {6, 2}, {0, 1})));
}

TEST(ShapeUtilTest, HasDegenerateDimensions) {
  EXPECT_TRUE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 1, 2})));
  EXPECT_TRUE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 1, 1})));
  EXPECT_FALSE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 3, 5})));
  EXPECT_FALSE(
      ShapeUtil::HasDegenerateDimensions(ShapeUtil::MakeShape(F32, {3, 0, 5})));
}

TEST(ShapeUtilTest, PermuteDimensionsLayout) {
  std::vector<int64> layout(3);
  std::iota(layout.begin(), layout.end(), 0);
  do {
    Shape s = ShapeUtil::MakeShapeWithLayout(F32, {10, 100, 1000}, layout);
    SCOPED_TRACE(tensorflow::strings::StrCat("s=", ShapeUtil::HumanString(s)));

    std::vector<int64> permutation(3);
    std::iota(permutation.begin(), permutation.end(), 0);
    do {
      SCOPED_TRACE(tensorflow::strings::StrCat(
          "permutation=", tensorflow::str_util::Join(permutation, ",")));

      // TransposeIsBitcast takes the inverse of the permutation that
      // PermuteDimensions takes.
      EXPECT_TRUE(ShapeUtil::TransposeIsBitcast(
          s, ShapeUtil::PermuteDimensions(permutation, s),
          InversePermutation(permutation)));
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  } while (std::next_permutation(layout.begin(), layout.end()));
}

TEST(AlgebraicSimplifierTest, ReshapeIsBitcast_3x2x2_6x2_Dim0IsMostMinor) {
  EXPECT_FALSE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithLayout(F32, {3, 2, 2}, {0, 1, 2}),
      ShapeUtil::MakeShapeWithLayout(F32, {6, 2}, {0, 1})));
}

TEST(AlignmentTest, AlignLayoutsWithoutTrivialDimensions) {
  Shape input = ShapeUtil::MakeShapeWithLayout(xla::F32, {3, 8, 5, 7, 11},
                                               {3, 2, 1, 0, 4});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {4, 3, 2, 7, 5, 11}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(4, 3, 2, 1, 0, 5));
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(input, aligned_shape.value()));

  aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {3, 2, 4, 35, 11}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(3, 2, 1, 0, 4));
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(input, aligned_shape.value()));
}

TEST(AlignmentTest, AlignLayoutsWithTrivialDimensions) {
  Shape input =
      ShapeUtil::MakeShapeWithLayout(xla::F32, {1, 3, 8, 1, 5, 7, 1, 11, 1, 1},
                                     {5, 0, 4, 2, 1, 3, 6, 7, 9, 8});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {1, 4, 1, 3, 2, 7, 5, 11, 1}));
  EXPECT_TRUE(aligned_shape);
  EXPECT_THAT(aligned_shape.value().layout().minor_to_major(),
              ElementsAre(6, 5, 4, 3, 1, 7, 0, 2, 8));
  EXPECT_TRUE(ShapeUtil::ReshapeIsBitcast(input, aligned_shape.value()));
}

// A test case where the consecutive elements of the input shape belonging to
// the same layout part are not in descending order.
TEST(AlignmentTest, AlignLayoutsWithoutTrivialDimensionsWrongInputLayout) {
  // Same physical layout as in AlignLayoutsWithoutTrivialDimensions, except
  // that the first two dimension numbers are exchanged.
  Shape input = ShapeUtil::MakeShapeWithLayout(xla::F32, {3, 8, 5, 7, 11},
                                               {2, 3, 1, 0, 4});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {4, 3, 2, 7, 5, 11}));
  EXPECT_FALSE(aligned_shape);
}

// A test case where the physical layout of the input shape does not place all
// dimensions that belong to the same alignment part consecutively.
TEST(AlignmentTest,
     AlignLayoutsWithoutTrivialDimensionsNonConsecutiveAlignmentPart) {
  Shape input = ShapeUtil::MakeShapeWithLayout(xla::F32, {3, 8, 5, 7, 11},
                                               {3, 2, 1, 0, 4});
  auto aligned_shape = ShapeUtil::AlignLayouts(
      input, ShapeUtil::MakeShape(xla::F32, {4, 3, 2, 5, 77}));
  EXPECT_FALSE(aligned_shape);
}

}  // namespace
}  // namespace xla
