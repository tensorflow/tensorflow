/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/layout_util.h"

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/test_helpers.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace {

class LayoutUtilTest : public ::testing::Test {
 protected:
  Shape MakeShapeWithLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dimensions,
      absl::Span<const int64_t> minor_to_major,
      absl::Span<const DimLevelType> dim_level_types = {}) {
    Shape shape = ShapeUtil::MakeShape(element_type, dimensions);
    *shape.mutable_layout() =
        LayoutUtil::MakeLayout(minor_to_major, dim_level_types);
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

TEST_F(LayoutUtilTest, CopyLayoutDenseArray) {
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

TEST_F(LayoutUtilTest, CopyLayoutCSRArray) {
  Shape src =
      MakeShapeWithLayout(F32, {2, 3}, {1, 0}, {DIM_DENSE, DIM_COMPRESSED});
  Shape dst = MakeShapeWithLayout(F32, {2, 3}, {0, 1});

  EXPECT_TRUE(LayoutUtil::IsSparseArray(src));
  EXPECT_FALSE(LayoutUtil::IsSparseArray(dst));

  EXPECT_TRUE(LayoutUtil::IsCSRArray(src));
  EXPECT_FALSE(LayoutUtil::IsCSRArray(dst));

  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_TRUE(LayoutUtil::IsCSRArray(dst));

  // Should work if destination has no layout.
  dst.clear_layout();
  EXPECT_FALSE(LayoutUtil::IsCSRArray(dst));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_TRUE(LayoutUtil::IsCSRArray(dst));

  // Convert dst to a CSC array with dim 0 minor layout.
  *dst.mutable_layout()->mutable_minor_to_major() = {0, 1};
  EXPECT_TRUE(LayoutUtil::IsCSCArray(dst));
  EXPECT_FALSE(LayoutUtil::IsCSRArray(dst));

  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  *src.mutable_layout()->mutable_physical_shape() = ShapeUtil::MakeTupleShape({
      ShapeUtil::MakeShapeWithDenseLayout(U32, {2}, {0}, {Tile({100})}),
      ShapeUtil::MakeShapeWithDenseLayout(U32, {4}, {0}, {Tile({100})}),
      ShapeUtil::MakeShapeWithDenseLayout(F32, {4}, {0}, {Tile({100})}),
  });
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  dst.clear_layout();
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));

  // If source is cleared, then destination should be cleared.
  src.clear_layout();
  EXPECT_FALSE(LayoutUtil::IsCSRArray(src));
  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_TRUE(dst.has_layout());
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_FALSE(dst.has_layout());
  EXPECT_FALSE(LayoutUtil::IsCSRArray(dst));
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
  EXPECT_THAT(status.message(),
              ::testing::ContainsRegex("cannot copy layout from shape"));
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
  EXPECT_THAT(status.message(),
              ::testing::ContainsRegex("cannot copy layout from shape"));
}

TEST_F(LayoutUtilTest, CopyLayoutBogusLayout) {
  Shape src = ShapeUtil::MakeShape(F32, {2, 3});
  Shape dst = ShapeUtil::MakeShape(F32, {2, 3});
  // Set layout to invalid value.
  *src.mutable_layout() = LayoutUtil::MakeLayout({1, 2, 3, 4});

  auto status = LayoutUtil::CopyLayoutBetweenShapes(src, &dst);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(), ::testing::ContainsRegex(
                                    "layout minor_to_major field contains .* "
                                    "elements, but shape is rank"));
}

TEST_F(LayoutUtilTest, CopyTokenLayout) {
  Shape src = ShapeUtil::MakeTokenShape();
  Shape dst = ShapeUtil::MakeTokenShape();

  // Layouts are trivially the same for token types and copying layouts should
  // be a nop.
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyOpaqueLayout) {
  Shape src = ShapeUtil::MakeOpaqueShape();
  Shape dst = ShapeUtil::MakeOpaqueShape();

  // Layouts are trivially the same for opaque types and copying layouts should
  // be a nop.
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
}

TEST_F(LayoutUtilTest, CopyTupleLayoutWithTokenAndOpaque) {
  Shape src = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {0, 1}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}), ShapeUtil::MakeTokenShape(),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeOpaqueShape(), MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {0, 2, 1})})});
  Shape dst = ShapeUtil::MakeTupleShape(
      {MakeShapeWithLayout(F32, {2, 3}, {1, 0}),
       MakeShapeWithLayout(F32, {42, 123}, {1, 0}), ShapeUtil::MakeTokenShape(),
       ShapeUtil::MakeTupleShape(
           {ShapeUtil::MakeOpaqueShape(), MakeShapeWithLayout(F32, {}, {}),
            MakeShapeWithLayout(F32, {1, 2, 3}, {1, 2, 0})})});

  EXPECT_FALSE(LayoutUtil::LayoutsInShapesEqual(src, dst));
  EXPECT_IS_OK(LayoutUtil::CopyLayoutBetweenShapes(src, &dst));
  EXPECT_TRUE(LayoutUtil::LayoutsInShapesEqual(src, dst));
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

TEST_F(LayoutUtilTest, ClearLayoutOpaqueAndToken) {
  // Opaque and token types trivially have layouts.
  for (Shape shape :
       {ShapeUtil::MakeOpaqueShape(), ShapeUtil::MakeTokenShape()}) {
    EXPECT_TRUE(LayoutUtil::HasLayout(shape));
    LayoutUtil::ClearLayout(&shape);
    EXPECT_TRUE(LayoutUtil::HasLayout(shape));
  }
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

TEST_F(LayoutUtilTest, DefaultLayoutGettersMajorToMinor) {
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

TEST_F(LayoutUtilTest, MakeDescending) {
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeDescendingLayout(5),
                                LayoutUtil::MakeLayout({4, 3, 2, 1, 0})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeDescendingLayout(1),
                                LayoutUtil::MakeLayout({0})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeDescendingLayout(0),
                                LayoutUtil::MakeLayout({})));
}

TEST_F(LayoutUtilTest, MakeAscending) {
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeAscendingLayout(5),
                                LayoutUtil::MakeLayout({0, 1, 2, 3, 4})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeAscendingLayout(1),
                                LayoutUtil::MakeLayout({0})));
  EXPECT_TRUE(LayoutUtil::Equal(LayoutUtil::MakeAscendingLayout(0),
                                LayoutUtil::MakeLayout({})));
}

TEST_F(LayoutUtilTest, HumanStringWithTiling) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3, 4}, {0, 1, 2});
  Tile* tile;

  // No tiling.
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape), "f32[2,3,4]{0,1,2}");

  // 2D tile.
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(512);
  tile->add_dimensions(1024);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "f32[2,3,4]{0,1,2:T(512,1024)}");

  // 1D tile.
  shape.mutable_layout()->clear_tiles();
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(512);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "f32[2,3,4]{0,1,2:T(512)}");

  // 2 tiles.
  shape = ShapeUtil::MakeShapeWithDenseLayout(BF16, {2, 3, 4}, {1, 2, 0});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(16);
  tile->add_dimensions(256);
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(2);
  tile->add_dimensions(1);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "bf16[2,3,4]{1,2,0:T(16,256)(2,1)}");

  // PRED with element size of 8 bits.
  shape = ShapeUtil::MakeShapeWithDenseLayout(PRED, {8, 8, 8}, {0, 2, 1});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(8);
  tile->add_dimensions(128);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "pred[8,8,8]{0,2,1:T(8,128)}");

  // PRED with element size of 32 bits.
  shape.mutable_layout()->clear_tiles();
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(8);
  tile->add_dimensions(128);
  shape.mutable_layout()->set_element_size_in_bits(32);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "pred[8,8,8]{0,2,1:T(8,128)E(32)}");

  // No tile. PRED with element size of 32 bits.
  shape.mutable_layout()->clear_tiles();
  shape.mutable_layout()->set_element_size_in_bits(32);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "pred[8,8,8]{0,2,1:E(32)}");

  // Tile with negative dimension size for combining dimensions.
  shape = ShapeUtil::MakeShapeWithDenseLayout(BF16, {2, 3, 1004}, {2, 1, 0});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(2);
  tile->add_dimensions(Tile::kCombineDimension);
  tile->add_dimensions(128);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "bf16[2,3,1004]{2,1,0:T(2,*,128)}");

  // Tile with two negative dimensions.
  shape =
      ShapeUtil::MakeShapeWithDenseLayout(BF16, {8, 2, 3, 1004}, {3, 2, 1, 0});
  tile = shape.mutable_layout()->add_tiles();
  tile->add_dimensions(2);
  tile->add_dimensions(Tile::kCombineDimension);
  tile->add_dimensions(Tile::kCombineDimension);
  tile->add_dimensions(128);
  EXPECT_EQ(ShapeUtil::HumanStringWithLayout(shape),
            "bf16[8,2,3,1004]{3,2,1,0:T(2,*,*,128)}");
}

TEST_F(LayoutUtilTest, ValidateLayout_ValidArrayLayout) {
  Shape shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {2, 3}, {0, 1});
  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_TRUE(status.ok());
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status.ok());
}

TEST_F(LayoutUtilTest, ValidateLayout_InvalidArrayLayout) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2});
  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("layout minor_to_major field "
                                   "contains 3 elements, but shape is rank 2"));
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("layout minor_to_major field "
                                   "contains 3 elements, but shape is rank 2"));
}

TEST_F(LayoutUtilTest, ValidateLayout_InvalidDimLevelTypes) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1});
  shape.mutable_layout()->add_dim_level_type(DIM_DENSE);
  shape.mutable_layout()->add_dim_level_type(DIM_DENSE);
  shape.mutable_layout()->add_dim_level_type(DIM_DENSE);
  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("layout dim_level_types field "
                                   "contains 3 elements, but shape is rank 2"));
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("layout dim_level_types field "
                                   "contains 3 elements, but shape is rank 2"));
}

TEST_F(LayoutUtilTest, ValidateLayout_MissingArrayLayout) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  LayoutUtil::ClearLayout(&shape);
  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("shape f32[2,3] does not have a layout"));
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status.ok());
}

TEST_F(LayoutUtilTest, ValidateLayout_Sparse) {
  Shape shape = ShapeUtil::MakeShape(F32, {2, 3});
  *shape.mutable_layout() = LayoutUtil::MakeLayout(
      {1, 0}, {DIM_DENSE, DIM_COMPRESSED}, {}, {}, {Tile({10, 10})});
  EXPECT_THAT(LayoutUtil::ValidateLayoutInShape(shape),
              tsl::testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  ::testing::HasSubstr(
                      "layout has tiles, but the shape is a sparse array")));
  shape.mutable_layout()->clear_tiles();
  EXPECT_THAT(LayoutUtil::ValidateLayoutInShape(shape), tsl::testing::IsOk());
  *shape.mutable_layout()->mutable_physical_shape() =
      ShapeUtil::MakeShape(F32, {6});
  EXPECT_THAT(LayoutUtil::ValidateLayoutInShape(shape), tsl::testing::IsOk());
  *shape.mutable_layout()
       ->mutable_physical_shape()
       ->mutable_layout()
       ->mutable_physical_shape() = ShapeUtil::MakeShape(S32, {10});
  EXPECT_THAT(
      LayoutUtil::ValidateLayoutInShape(shape),
      tsl::testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          ::testing::HasSubstr(
              "layout has a physical_shape, but is not a sparse array")));
  shape.mutable_layout()->mutable_physical_shape()->clear_layout();
  shape.mutable_layout()->clear_dim_level_types();
  EXPECT_THAT(
      LayoutUtil::ValidateLayoutInShape(shape),
      tsl::testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          ::testing::HasSubstr(
              "layout has a physical_shape, but is not a sparse array")));
  *shape.mutable_layout() =
      LayoutUtil::MakeLayout({1, 0}, {DIM_DENSE, DIM_DENSE}, {true, false});
  EXPECT_THAT(LayoutUtil::ValidateLayoutInShape(shape),
              tsl::testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  ::testing::HasSubstr("layout dimension 1 has invalid level "
                                       "encoding DIM_DENSE, non-unique")));
}

TEST_F(LayoutUtilTest, ValidateLayout_TupleSubshapesWithMissingLayouts) {
  Shape sub_1_1_1 = ShapeUtil::MakeShape(F32, {1, 2});
  Shape sub_1_1 = ShapeUtil::MakeTupleShape({sub_1_1_1});
  Shape sub_1_2 = ShapeUtil::MakeShape(F32, {1, 2});
  LayoutUtil::ClearLayout(&sub_1_2);
  Shape sub_1 = ShapeUtil::MakeTupleShape({sub_1_1, sub_1_2});
  Shape sub_2_1 = ShapeUtil::MakeShape(F32, {9});
  LayoutUtil::ClearLayout(&sub_2_1);
  Shape sub_2 = ShapeUtil::MakeTupleShape({sub_2_1});
  Shape shape = ShapeUtil::MakeTupleShape({sub_1, sub_2});

  auto status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/false);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("shape f32[1,2] does not have a layout"));
  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_TRUE(status.ok());

  // Add invalid layout on one of sub-shapes.
  *shape.mutable_tuple_shapes(1)->mutable_tuple_shapes(0)->mutable_layout() =
      LayoutUtil::MakeLayout({0, 2, 3});

  status =
      LayoutUtil::ValidateLayoutInShape(shape, /*allow_missing_layouts=*/true);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("layout minor_to_major field "
                                   "contains 3 elements, but shape is rank 1"));
}

TEST_F(LayoutUtilTest, MoveDimToMajor) {
  const Layout layout = LayoutUtil::MakeLayout({2, 1, 0});
  Layout new_layout = LayoutUtil::MoveDimToMajor(layout, 0);
  EXPECT_EQ(new_layout, layout);

  new_layout = LayoutUtil::MoveDimToMajor(layout, 1);
  EXPECT_EQ(new_layout, LayoutUtil::MakeLayout({2, 0, 1}));
}

TEST_F(LayoutUtilTest, StridesIsMajorToMinor) {
  std::vector<int64_t> byte_strides = {3960, 440, 44, 4};
  EXPECT_TRUE(LayoutUtil::ByteStridesIsMajorToMinor(
      byte_strides, {8, 9, 10, 11}, PrimitiveType::F32));
}

TEST_F(LayoutUtilTest, StridesNotMajorToMinorInnerMostStrideIncorrect) {
  std::vector<int64_t> byte_strides = {1880, 220, 22, 2};
  EXPECT_FALSE(LayoutUtil::ByteStridesIsMajorToMinor(
      byte_strides, {8, 9, 10, 11}, PrimitiveType::F32));
}

TEST_F(LayoutUtilTest, StridesNotMajorToMinor) {
  std::vector<int64_t> byte_strides = {1880, 440, 44, 4};
  EXPECT_FALSE(LayoutUtil::ByteStridesIsMajorToMinor(
      byte_strides, {8, 9, 10, 11}, PrimitiveType::F32));
}

TEST_F(LayoutUtilTest, HasCustomElementSizeInBits) {
  Shape shape = ShapeUtil::MakeShape(F32, {1, 2});
  EXPECT_FALSE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeShape(F32, {1, 2});
  shape.mutable_layout()->set_element_size_in_bits(0);
  EXPECT_FALSE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeShape(F32, {1, 2});
  shape.mutable_layout()->set_element_size_in_bits(32);
  EXPECT_TRUE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 2}),
                                  ShapeUtil::MakeShape(F32, {1, 2})}),
       ShapeUtil::MakeShape(F32, {1, 2})});
  EXPECT_FALSE(LayoutUtil::HasCustomElementSizeInBits(shape));

  shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 2}),
                                  ShapeUtil::MakeShape(F32, {1, 2})}),
       ShapeUtil::MakeShape(F32, {1, 2})});
  ShapeUtil::GetMutableSubshape(&shape, {0, 1})
      ->mutable_layout()
      ->set_element_size_in_bits(32);
  EXPECT_TRUE(LayoutUtil::HasCustomElementSizeInBits(shape));
}

TEST_F(LayoutUtilTest, MaxSplitSize) {
  Shape shape = ShapeUtil::MakeShape(F32, {150, 200, 100});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2})
                                .add_split_configs(SplitConfig(0, {30}))
                                .add_split_configs(SplitConfig(1, {40, 130}));

  EXPECT_EQ(LayoutUtil::MaxSplitSize(shape, 0), 150);
  EXPECT_EQ(LayoutUtil::MaxSplitSize(shape, 1), 90);
  EXPECT_EQ(LayoutUtil::MaxSplitSize(shape, 2), 70);
}

TEST_F(LayoutUtilTest, MaxElementsInPerSplit) {
  Shape shape = ShapeUtil::MakeShape(F32, {150, 200, 100});
  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2});
  EXPECT_EQ(LayoutUtil::MaxElementsInPerSplit(shape), 150 * 200 * 100);

  *shape.mutable_layout() = LayoutUtil::MakeLayout({0, 1, 2})
                                .add_split_configs(SplitConfig(0, {30}))
                                .add_split_configs(SplitConfig(1, {40, 130}));
  EXPECT_EQ(LayoutUtil::MaxElementsInPerSplit(shape), 150 * 90 * 70);
}

}  // namespace
}  // namespace xla
