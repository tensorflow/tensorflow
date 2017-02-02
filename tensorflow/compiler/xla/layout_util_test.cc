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

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"

#include "tensorflow/compiler/xla/legacy_flags/layout_util_flags.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class LayoutUtilTest : public ::testing::Test {
 protected:
  Shape MakeShapeWithLayout(PrimitiveType element_type,
                            tensorflow::gtl::ArraySlice<int64> dimensions,
                            tensorflow::gtl::ArraySlice<int64> minor_to_major) {
    Shape shape = ShapeUtil::MakeShape(element_type, dimensions);
    *shape.mutable_layout() = LayoutUtil::MakeLayout(minor_to_major);
    return shape;
  }
};

TEST_F(LayoutUtilTest, TupleLayoutComparison) {
  Shape shape =
      ShapeUtil::MakeTupleShape({MakeShapeWithLayout(F32, {2, 3}, {0, 1})});
  Shape other_shape =
      ShapeUtil::MakeTupleShape({MakeShapeWithLayout(F32, {2, 2}, {0, 1})});

  Shape tuple0 = ShapeUtil::MakeTupleShape({});
  Shape tuple1 = ShapeUtil::MakeTupleShape({shape});
  Shape tuple2 = ShapeUtil::MakeTupleShape({shape, shape});

  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple0, tuple0));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple0, tuple1));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple0, tuple2));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple1, tuple0));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple2, tuple0));

  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple1, tuple1));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple1, tuple2));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(tuple2, tuple1));

  Shape other_tuple2 = ShapeUtil::MakeTupleShape({shape, other_shape});
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple2, tuple2));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(tuple2, other_tuple2));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(other_tuple2, tuple2));
}

TEST_F(LayoutUtilTest, CopyLayoutArray) {
  Shape src = MakeShapeWithLayout(F32, {2, 3}, {0, 1});
  Shape dst = MakeShapeWithLayout(F32, {2, 3}, {1, 0});

  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));

  // Should work if destination has no layout.
  dst.clear_layout();
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));

  // If source is cleared, then destination should be cleared.
  src.clear_layout();
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_TRUE(dst.has_layout());
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_FALSE(dst.has_layout());
}

TEST_F(LayoutUtilTest, CopyLayoutTuple) {
  Shape src = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {0, 1}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {0, 2, 1})})});
  Shape dst = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});

  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleSameRank) {
  Shape src = MakeShapeWithLayout(F32, {123, 42, 7}, {2, 0, 1});
  Shape dst = MakeShapeWithLayout(F32, {2, 3, 5}, {1, 0});
  ASSERT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleDifferentRank) {
  Shape src = MakeShapeWithLayout(F32, {123, 42, 7}, {2, 0, 1});
  Shape dst = MakeShapeWithLayout(F32, {2, 3}, {1, 0});
  auto status = LayoutUtil::CopyLayoutBetweenShapes(src, &dst);
  EXPECT_FALSE(status.ok());
  EXPECT_MATCH(status.error_message(),
               testing::ContainsRegex("cannot copy layout from shape"));
}

TEST_F(LayoutUtilTest, CopyLayoutNotCompatibleTuple) {
  Shape src =
      ShapeUtil::MakeTupleShape({MakeShapeWithLayout(F32, {2, 3}, {0, 1}),
                                 MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
                                 ShapeUtil::MakeTupleShape({MakeShapeWithLayout(
                                     F32, {1, 2, 3}, {0, 2, 1})})});
  Shape dst = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});

  auto status = LayoutUtil::CopyLayoutBetweenShapes(src, &dst);
  EXPECT_FALSE(status.ok());
  EXPECT_MATCH(status.error_message(),
               testing::ContainsRegex("cannot copy layout from shape"));
}

TEST_F(LayoutUtilTest, CopyLayoutBogusLayout) {
  Shape src = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst = ShapeUtil::MakeShape(F32, {2, 3});
  // Set layout to invalid value.
  *src.mutable_layout() = LayoutUtil::MakeLayout({1, 2, 3, 4});

  auto status = LayoutUtil::CopyLayoutBetweenShapes(src, &dst);
  EXPECT_FALSE(status.ok());
  EXPECT_MATCH(status.error_message(),
               testing::ContainsRegex("layout minor_to_major field contains .* "
                                      "elements, but shape is rank"));
}

TEST_F(LayoutUtilTest, ClearLayoutTuple) {
  Shape shape = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});
  EXPECT_TRUE(LayoutUtil::HasLayout(shape));
  EXPECT_TRUE(shape.tuple_shapes(0).has_layout());
  EXPECT_TRUE(shape.tuple_shapes(2).tuple_shapes(1).has_layout());

  LayoutUtil::ClearLayout(&shape);

  EXPECT_FALSE(LayoutUtil::HasLayout(shape));
  EXPECT_FALSE(shape.tuple_shapes(0).has_layout());
  EXPECT_FALSE(shape.tuple_shapes(2).tuple_shapes(1).has_layout());
}

TEST_F(LayoutUtilTest, SetToDefaultLayoutTuple) {
  Shape shape = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3, 4}, {1, 0, 2}),
       MakeShapeWithLayout(F32, {42, 123, 7}, {1, 2, 0}),
       ShapeUtil::MakeTupleShape(
           {MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3, 4}, {3, 1, 2, 0})})});
  EXPECT_FALSE(LayoutUtil::Equal(shape.tuple_shapes(0).layout(),
                                 shape.tuple_shapes(1).layout()));
  LayoutUtil::SetToDefaultLayout(&shape);
  EXPECT_TRUE(LayoutUtil::Equal(shape.tuple_shapes(0).layout(),
                                shape.tuple_shapes(1).layout()));
  EXPECT_TRUE(LayoutUtil::Equal(
      LayoutUtil::GetDefaultLayoutForShape(shape.tuple_shapes(0)),
      shape.tuple_shapes(1).layout()));
}

TEST_F(LayoutUtilTest, IsPadded) {
  Shape shape_without_layout = ShapeUtil::MakeShape(F32, {2, 3, 4});
  LayoutUtil::ClearLayout(&shape_without_layout);
  EXPECT_FALSE(LayoutUtil::IsPadded(shape_without_layout));

  Shape shape_with_layout = ShapeUtil::MakeShape(F32, {2, 3, 4});
  LayoutUtil::SetToDefaultLayout(&shape_with_layout);
  EXPECT_FALSE(LayoutUtil::IsPadded(shape_with_layout));

  // Add padding equal to the dimension sizes. In this case the padding is a
  // nop.
  Shape shape_with_degenerate_padding = ShapeUtil::MakeShape(F32, {2, 3, 4});
  shape_with_degenerate_padding.mutable_layout()->add_padded_dimensions(2);
  shape_with_degenerate_padding.mutable_layout()->add_padded_dimensions(3);
  shape_with_degenerate_padding.mutable_layout()->add_padded_dimensions(4);
  EXPECT_FALSE(LayoutUtil::IsPadded(shape_with_degenerate_padding));

  Shape shape_with_padding = ShapeUtil::MakeShape(F32, {2, 3, 4});
  shape_with_padding.mutable_layout()->add_padded_dimensions(2);
  shape_with_padding.mutable_layout()->add_padded_dimensions(14);
  shape_with_padding.mutable_layout()->add_padded_dimensions(42);
  EXPECT_TRUE(LayoutUtil::IsPadded(shape_with_padding));
}

TEST_F(LayoutUtilTest, DefaultLayoutGettersMajorToMinor) {
  // Test that LayoutUtil returns expected layouts when the xla_default_layout
  // flag is set to kMajorToMinor.
  legacy_flags::LayoutUtilFlags* flags = legacy_flags::GetLayoutUtilFlags();
  flags->xla_default_layout = xla::legacy_flags::DefaultLayout{
      .dimension_order =
          legacy_flags::DefaultLayout::DimensionOrder::kMajorToMinor};

  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({1, 0}),
                                LayoutUtil::GetDefaultLayoutForR2()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({2, 1, 0}),
                                LayoutUtil::GetDefaultLayoutForR3()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({3, 2, 1, 0}),
                                LayoutUtil::GetDefaultLayoutForR4()));
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeLayout({4, 3, 2, 1, 0}),
                        LayoutUtil::GetDefaultLayoutForShape(
                            ShapeUtil::MakeShape(F32, {10, 20, 30, 15, 25}))));
}

TEST_F(LayoutUtilTest, DefaultLayoutGettersMinorToMajor) {
  // Test that LayoutUtil returns expected layouts when the xla_default_layout
  // flag is set to kMinorToMajor.
  legacy_flags::LayoutUtilFlags* flags = legacy_flags::GetLayoutUtilFlags();
  flags->xla_default_layout = xla::legacy_flags::DefaultLayout{
      .dimension_order =
          legacy_flags::DefaultLayout::DimensionOrder::kMinorToMajor};

  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1}),
                                LayoutUtil::GetDefaultLayoutForR2()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1, 2}),
                                LayoutUtil::GetDefaultLayoutForR3()));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1, 2, 3}),
                                LayoutUtil::GetDefaultLayoutForR4()));
  EXPECT_TRUE(
      LayoutUtil::Equal(LayoutUtil::MakeLayout({0, 1, 2, 3, 4}),
                        LayoutUtil::GetDefaultLayoutForShape(
                            ShapeUtil::MakeShape(F32, {10, 20, 30, 15, 25}))));
}

}  // namespace
}  // namespace xla
