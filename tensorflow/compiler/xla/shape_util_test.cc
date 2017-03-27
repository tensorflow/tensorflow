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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

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
  Shape actual = ShapeUtil::ParseShapeString(shape_string).ValueOrDie();
  Shape expected = ShapeUtil::MakeShape(F32, {123, 456});
  ASSERT_TRUE(ShapeUtil::Equal(expected, actual))
      << "expected: " << ShapeUtil::HumanString(expected)
      << "actual:   " << ShapeUtil::HumanString(actual);
}

TEST(ShapeUtilTest, CompatibleIdenticalShapes) {
  Shape shape1 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape2 = ShapeUtil::MakeShape(F32, {3, 2});
  ASSERT_TRUE(ShapeUtil::Compatible(shape1, shape2));
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

TEST(ShapeUtilTest, IncompatibleDifferentElementShapes) {
  Shape shape_1 = ShapeUtil::MakeShape(F32, {3, 2});
  Shape shape_2 = ShapeUtil::MakeShape(PRED, {3, 2});
  EXPECT_FALSE(ShapeUtil::Compatible(shape_1, shape_2));
}

TEST(ShapeUtilTest, CompatibleTuples) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  EXPECT_TRUE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithSwappedElements) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 2}), ShapeUtil::MakeShape(PRED, {4, 5})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithDifferentPrimitiveType) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(S32, {3, 2})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, IncompatibleTuplesWithDifferentDimensions) {
  Shape tuple1 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {3, 2})});
  Shape tuple2 = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {4, 5}), ShapeUtil::MakeShape(F32, {4, 2})});
  EXPECT_FALSE(ShapeUtil::Compatible(tuple1, tuple2));
}

TEST(ShapeUtilTest, EmptyLayoutEqualsMissingLayout) {
  // A shape with a missing layout should be equal to a shape with an empty
  // layout.
  Shape scalar1 = ShapeUtil::MakeShape(F32, {});
  Shape scalar2 = ShapeUtil::MakeShape(F32, {});

  EXPECT_TRUE(ShapeUtil::Equal(scalar1, scalar2));

  scalar1.clear_layout();    // Remove layout field.
  scalar2.mutable_layout();  // Create empty layout field.

  EXPECT_TRUE(ShapeUtil::Equal(scalar1, scalar2));
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

TEST(ShapeUtilTest, ScalarUnpopulatedLayoutEqualsScalarLayout) {
  Shape scalar_unpopulated = ShapeUtil::MakeShape(F32, {});
  scalar_unpopulated.clear_layout();
  ASSERT_FALSE(scalar_unpopulated.has_layout())
      << ShapeUtil::HumanStringWithLayout(scalar_unpopulated);

  const Shape scalar_populated = ShapeUtil::MakeShapeWithLayout(F32, {}, {});
  ASSERT_TRUE(scalar_populated.has_layout())
      << ShapeUtil::HumanStringWithLayout(scalar_populated);

  EXPECT_TRUE(ShapeUtil::Equal(scalar_unpopulated, scalar_populated));
}

TEST(ShapeUtilTest, ByteSizeOfWithoutPadding) {
  EXPECT_EQ(4, ShapeUtil::ByteSizeOfPrimitiveType(F32));
  EXPECT_EQ(4, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F32, {})));
  EXPECT_EQ(800, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F32, {10, 20})));

  EXPECT_EQ(8, ShapeUtil::ByteSizeOfPrimitiveType(F64));
  EXPECT_EQ(8, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F64, {})));
  EXPECT_EQ(1600, ShapeUtil::ByteSizeOf(ShapeUtil::MakeShape(F64, {10, 20})));
}

TEST(ShapeUtilTest, ByteSizeOfWithPadding) {
  EXPECT_EQ(4, ShapeUtil::ByteSizeOfPrimitiveType(F32));
  Shape shape = ShapeUtil::MakeShape(F32, {10, 20});
  EXPECT_EQ(800, ShapeUtil::ByteSizeOf(shape));

  shape.mutable_layout()->add_padded_dimensions(15);
  shape.mutable_layout()->add_padded_dimensions(21);
  EXPECT_EQ(15 * 21 * 4, ShapeUtil::ByteSizeOf(shape));
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

TEST(ShapeUtilTest, HasZeroElements) {
  EXPECT_EQ(false, ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {})));
  EXPECT_EQ(true, ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {0})));
  EXPECT_EQ(false, ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {1})));
  EXPECT_EQ(false,
            ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {1, 1})));
  EXPECT_EQ(false, ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {2})));
  EXPECT_EQ(false,
            ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {2, 1})));
  EXPECT_EQ(false,
            ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {3, 5})));
  EXPECT_EQ(true,
            ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {3, 0, 5})));
  EXPECT_EQ(true,
            ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {0, 3, 0})));
  EXPECT_EQ(false,
            ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {1, 3, 5})));
  EXPECT_EQ(false,
            ShapeUtil::HasZeroElements(ShapeUtil::MakeShape(S32, {13, 17})));
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

TEST(ShapeUtilTest, HumanString) {
  Shape opaque = ShapeUtil::MakeOpaqueShape();
  Shape scalar = ShapeUtil::MakeShape(F32, {});
  Shape matrix = ShapeUtil::MakeShape(U32, {1, 2});
  Shape matrix2 = ShapeUtil::MakeShapeWithLayout(S32, {3, 4}, {0, 1});
  Shape tuple = ShapeUtil::MakeTupleShape({opaque, scalar, matrix, matrix2});
  Shape nested_tuple = ShapeUtil::MakeTupleShape({tuple, matrix});

  EXPECT_EQ("opaque[]", ShapeUtil::HumanString(opaque));
  EXPECT_EQ("f32[]", ShapeUtil::HumanString(scalar));
  EXPECT_EQ("u32[1,2]", ShapeUtil::HumanString(matrix));
  EXPECT_EQ("s32[3,4]", ShapeUtil::HumanString(matrix2));
  EXPECT_EQ("(opaque[], f32[], u32[1,2], s32[3,4])",
            ShapeUtil::HumanString(tuple));
  EXPECT_EQ("((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2])",
            ShapeUtil::HumanString(nested_tuple));

  EXPECT_EQ("opaque[]", ShapeUtil::HumanStringWithLayout(opaque));
  EXPECT_EQ("f32[]", ShapeUtil::HumanStringWithLayout(scalar));
  EXPECT_EQ("u32[1,2]{1,0}", ShapeUtil::HumanStringWithLayout(matrix));
  EXPECT_EQ("s32[3,4]{0,1}", ShapeUtil::HumanStringWithLayout(matrix2));
  EXPECT_EQ("(opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1})",
            ShapeUtil::HumanStringWithLayout(tuple));
  EXPECT_EQ("((opaque[], f32[], u32[1,2]{1,0}, s32[3,4]{0,1}), u32[1,2]{1,0})",
            ShapeUtil::HumanStringWithLayout(nested_tuple));

  ProgramShape prog = ShapeUtil::MakeProgramShape(
      {opaque, scalar, matrix, matrix2, tuple, nested_tuple}, nested_tuple);
  EXPECT_EQ(
      "((unknown): opaque[], "
      "(unknown): f32[], "
      "(unknown): u32[1,2], "
      "(unknown): s32[3,4], "
      "(unknown): (opaque[], f32[], u32[1,2], s32[3,4]), "
      "(unknown): ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2])) -> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2])",
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
      "nested_tuple: ((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2])) -> "
      "((opaque[], f32[], u32[1,2], s32[3,4]), u32[1,2])",
      ShapeUtil::HumanString(prog));
}

TEST(ShapeUtilTest, ForEachSubshapeArray) {
  const Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  int calls = 0;
  EXPECT_IS_OK(ShapeUtil::ForEachSubshape(
      shape, [&calls, &shape](const Shape& subshape, const ShapeIndex& index) {
        EXPECT_EQ(&shape, &subshape);
        EXPECT_TRUE(index.empty());
        ++calls;
        return tensorflow::Status::OK();
      }));
  EXPECT_EQ(1, calls);
}

TEST(ShapeUtilTest, ForEachSubshapeNestedTuple) {
  const Shape shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {101}),
                                  ShapeUtil::MakeShape(PRED, {33})})});
  int calls = 0;
  EXPECT_IS_OK(ShapeUtil::ForEachSubshape(
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
        return tensorflow::Status::OK();
      }));
  EXPECT_EQ(5, calls);
}

TEST(ShapeUtilTest, ForEachMutableSubshapeNestedTuple) {
  Shape shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {42}),
       ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {101}),
                                  ShapeUtil::MakeShape(PRED, {33})})});
  int calls = 0;
  EXPECT_IS_OK(ShapeUtil::ForEachMutableSubshape(
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
        return tensorflow::Status::OK();
      }));
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

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1x1_to_1x1x1) {
  // All output dimensions should be unmodified. One of the input dimensions is
  // modified because the input rank is larger by one.
  EXPECT_EQ(3,
            ShapeUtil::DimensionsUnmodifiedByReshape(
                ShapeUtil::MakeShape(S32, {1, 1, 1, 1}),
                ShapeUtil::MakeShape(S32, {1, 1, 1}))
                .size());
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_1x1x1_to_1x1x1x1) {
  // All input dimensions should be unmodified. One of the output dimensions is
  // modified because the output rank is larger by one.
  EXPECT_EQ(3,
            ShapeUtil::DimensionsUnmodifiedByReshape(
                ShapeUtil::MakeShape(S32, {1, 1, 1}),
                ShapeUtil::MakeShape(S32, {1, 1, 1, 1}))
                .size());
}

TEST(ShapeUtilTest, DimensionsUnmodifiedByReshape_4x1x3x5x6x7_to_2x6x1x5x1x42) {
  // The only matching dimension is the one with size 5.
  // 4, 1, 3, 5, 6, 7
  //          |
  // 2, 6, 1, 5, 1, 42
  EXPECT_TRUE(
      ContainersEqual(ShapeUtil::DimensionsUnmodifiedByReshape(
                          ShapeUtil::MakeShape(S32, {4, 1, 3, 5, 6, 7}),
                          ShapeUtil::MakeShape(S32, {2, 6, 1, 5, 1, 42})),
                      std::vector<std::pair<int64, int64>>({{3, 3}})));
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

TEST(AlgebraicSimplifierTest, ReshapeIsBitcast_3x2x2_6x2_Dim0IsMostMinor) {
  EXPECT_FALSE(ShapeUtil::ReshapeIsBitcast(
      ShapeUtil::MakeShapeWithLayout(F32, {3, 2, 2}, {0, 1, 2}),
      ShapeUtil::MakeShapeWithLayout(F32, {6, 2}, {0, 1})));
}

}  // namespace
}  // namespace xla
