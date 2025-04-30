/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/layout.h"

#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/testlib/test.h"
#include "xla/layout_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::testing::ElementsAre;

TEST(Layout, ToStringForEmpty) { EXPECT_EQ(Layout().ToString(), "{}"); }

TEST(Layout, ToStringForMinorToMajorOnly) {
  EXPECT_EQ(Layout({1, 2, 0}).ToString(), "{1,2,0}");
}

TEST(Layout, ToStringForDimensionAttributes) {
  // If all dimensions are dense, the dimension attributes are omitted.
  EXPECT_EQ(Layout({0}).add_dim_level_type(DIM_DENSE).ToString(), "{0}");
  EXPECT_EQ(Layout({1, 0})
                .add_dim_level_type(DIM_DENSE)
                .add_dim_level_type(DIM_DENSE)
                .ToString(),
            "{1,0}");

  // Test other dimension level type abbreviations.
  EXPECT_EQ(Layout({0}).add_dim_level_type(DIM_COMPRESSED).ToString(),
            "{0:D(C)}");
  EXPECT_EQ(Layout({0}).add_dim_level_type(DIM_SINGLETON).ToString(),
            "{0:D(S)}");
  EXPECT_EQ(Layout({0}).add_dim_level_type(DIM_LOOSE_COMPRESSED).ToString(),
            "{0:D(H)}");

  // Test the ordered attribute.
  EXPECT_EQ(Layout({0})
                .add_dim_level_type(DIM_COMPRESSED)
                .add_dim_ordered(false)
                .ToString(),
            "{0:D(C~)}");

  // Test the unique attribute.
  EXPECT_EQ(Layout({0})
                .add_dim_level_type(DIM_COMPRESSED)
                .add_dim_unique(false)
                .ToString(),
            "{0:D(C+)}");

  // Test the combination of ordered and unique attributes.
  EXPECT_EQ(Layout({0})
                .add_dim_level_type(DIM_COMPRESSED)
                .add_dim_ordered(false)
                .add_dim_unique(false)
                .ToString(),
            "{0:D(C+~)}");

  // Test multiple dimension attributes.
  EXPECT_EQ(Layout({1, 0})
                .add_dim_level_type(DIM_DENSE)
                .add_dim_level_type(DIM_COMPRESSED)
                .ToString(),
            "{1,0:D(D,C)}");
}

TEST(Layout, ToStringForTiles) {
  EXPECT_EQ(Layout({3, 2, 1, 0}, {}, {}, {}, {Tile({42, 123}), Tile({4, 5})})
                .ToString(),
            "{3,2,1,0:T(42,123)(4,5)}");
}

TEST(Layout, ToStringForTileWithCombinedDimensions) {
  EXPECT_EQ(
      Layout(
          {3, 2, 1, 0}, {}, {}, {},
          {Tile({Tile::kCombineDimension, Tile::kCombineDimension, 42, 123})})
          .ToString(),
      "{3,2,1,0:T(*,*,42,123)}");
}

TEST(Layout, ToStringForTailPaddingAlignment) {
  EXPECT_EQ(Layout({3, 2, 1, 0})
                .set_tail_padding_alignment_in_elements(100)
                .ToString(),
            "{3,2,1,0:L(100)}");
}

TEST(Layout, ToStringForIndexPrimitiveType) {
  EXPECT_EQ(Layout({3, 2, 1, 0})
                .set_index_primitive_type(PrimitiveType::U32)
                .ToString(),
            "{3,2,1,0:#(u32)}");
}

TEST(Layout, ToStringForPointerPrimitiveType) {
  EXPECT_EQ(Layout({3, 2, 1, 0})
                .set_pointer_primitive_type(PrimitiveType::U16)
                .ToString(),
            "{3,2,1,0:*(u16)}");
}

TEST(Layout, ToStringForElementSize) {
  EXPECT_EQ(Layout({3, 2, 1, 0}).set_element_size_in_bits(42).ToString(),
            "{3,2,1,0:E(42)}");
}

TEST(Layout, ToStringForMemorySpace) {
  EXPECT_EQ(Layout({3, 2, 1, 0}).set_memory_space(3).ToString(),
            "{3,2,1,0:S(3)}");
}

TEST(Layout, ToStringForSplitConfigs) {
  EXPECT_EQ(Layout({0, 1})
                .add_split_configs(SplitConfig(0, {3}))
                .add_split_configs(SplitConfig(1, {0, 4}))
                .ToString(),
            "{0,1:SC(0:3)(1:0,4)}");
}

TEST(Layout, ToStringForPhysicalShape) {
  Layout layout({0, 1});
  *layout.mutable_physical_shape() = ShapeUtil::MakeShape(S32, {10, 20});
  EXPECT_EQ(layout.ToString(), "{0,1:P(s32[10,20]{1,0})}");
}

TEST(Layout, ToStringForDynamicShapeMetadataPrefixBytes) {
  EXPECT_EQ(
      Layout({0, 1}).set_dynamic_shape_metadata_prefix_bytes(123).ToString(),
      "{0,1:M(123)}");
}

TEST(Layout, ToStringForMutipleProperties) {
  EXPECT_EQ(Layout({3, 2, 1, 0}, {}, {}, {}, {Tile({42, 123}), Tile({4, 5})})
                .set_tail_padding_alignment_in_elements(100)
                .set_element_size_in_bits(42)
                .ToString(),
            "{3,2,1,0:T(42,123)(4,5)L(100)E(42)}");
}

TEST(Layout, StreamOut) {
  {
    std::ostringstream oss;
    oss << Tile({7, 8});
    EXPECT_EQ(oss.str(), "(7,8)");
  }

  {
    std::ostringstream oss;
    oss << Layout({0, 1, 2});
    EXPECT_EQ(oss.str(), "{0,1,2}");
  }
}

TEST(Layout, Equality) {
  EXPECT_EQ(Layout(), Layout());
  const std::vector<int64_t> empty_dims;
  EXPECT_EQ(Layout(empty_dims), Layout(empty_dims));
  EXPECT_EQ(Layout(), Layout(empty_dims));
  EXPECT_EQ(Layout({0, 1, 2, 3}), Layout({0, 1, 2, 3}));
  EXPECT_NE(Layout({0, 1, 2, 3}), Layout({0, 1, 2}));
  EXPECT_EQ(Layout({0, 1, 2}, {}, {}, {}, {Tile({42, 44})}),
            Layout({0, 1, 2}, {}, {}, {}, {Tile({42, 44})}));
  EXPECT_NE(Layout({0, 1, 2}, {}, {}, {}, {Tile({42, 44})}),
            Layout({0, 1, 2}, {}, {}, {}, {Tile({42, 45})}));
  EXPECT_NE(Layout({0, 1, 2}, {}, {}, {}, {Tile({42, 44})}),
            Layout({0, 1, 2, 3}));
  EXPECT_EQ(Layout({0, 1, 2}).set_element_size_in_bits(33),
            Layout({0, 1, 2}).set_element_size_in_bits(33));
  EXPECT_NE(Layout({0, 1, 2}).set_element_size_in_bits(33),
            Layout({0, 1, 2}).set_element_size_in_bits(7));
  EXPECT_EQ(Layout({0, 1, 2}).set_memory_space(3),
            Layout({0, 1, 2}).set_memory_space(3));
  EXPECT_NE(Layout({0, 1, 2}).set_memory_space(1),
            Layout({0, 1, 2}).set_memory_space(3));
  EXPECT_FALSE(Layout::Equal()(Layout({0, 1, 2}, {}, {}, {}, {Tile({42, 44})}),
                               Layout({0, 1, 2})));
  EXPECT_EQ(Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
            Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})));
  EXPECT_NE(Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
            Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {3})));
  EXPECT_TRUE(Layout::Equal().IgnoreTiles()(
      Layout({0, 1, 2}, {}, {}, {}, {Tile({42, 44})}), Layout({0, 1, 2})));
  EXPECT_FALSE(Layout::Equal()(
      Layout({0, 1, 2}, {}, {}, {}, {}, 1, PRIMITIVE_TYPE_INVALID,
             PRIMITIVE_TYPE_INVALID, 32),
      Layout({0, 1, 2}, {}, {}, {}, {}, 1, PRIMITIVE_TYPE_INVALID,
             PRIMITIVE_TYPE_INVALID, 1)));
  EXPECT_TRUE(Layout::Equal().IgnoreElementSize()(
      Layout({0, 1, 2}).set_element_size_in_bits(32),
      Layout({0, 1, 2}).set_element_size_in_bits(1)));
  EXPECT_TRUE(Layout::Equal().IgnoreMemorySpace()(
      Layout({0, 1, 2}).set_memory_space(1),
      Layout({0, 1, 2}).set_memory_space(3)));
  EXPECT_TRUE(Layout::Equal().IgnoreSplitConfigs()(
      Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {2})),
      Layout({0, 1, 2}).add_split_configs(SplitConfig(0, {3}))));
}

TEST(Layout, LayoutToFromProto) {
  // Round-trips a Layout through proto de/serialization.
  auto expect_unchanged = [](const Layout& layout) {
    auto layout_proto = layout.ToProto();
    auto from_proto_result = Layout::FromProto(layout_proto);
    TF_ASSERT_OK(from_proto_result);
    EXPECT_EQ(layout, from_proto_result.value());
  };

  expect_unchanged(Layout());
  expect_unchanged(Layout({1, 3, 2, 0}));
  expect_unchanged(Layout({0, 1}).set_element_size_in_bits(42));
  expect_unchanged(
      Layout({3, 2, 1, 0}, {}, {}, {}, {Tile({42, 123}), Tile({4, 5})}));
  expect_unchanged(Layout({1, 0}, {DIM_DENSE, DIM_COMPRESSED}, {}, {}, {}));
  expect_unchanged(
      Layout({1, 0}, {DIM_DENSE, DIM_COMPRESSED}, {}, {}, {}, 1,
             PRIMITIVE_TYPE_INVALID, PRIMITIVE_TYPE_INVALID, 0, 0, {},
             std::make_unique<Shape>(ShapeUtil::MakeShape(S32, {10, 10}))));
  expect_unchanged(Layout({0, 1}, {}, {}, {}, {Tile({123})})
                       .add_split_configs(SplitConfig(0, {3}))
                       .add_split_configs(SplitConfig(1, {0, 4})));
}

TEST(Layout, DimensionIsUniqueByDefault) {
  Layout layout({0, 1});
  layout.add_dim_level_type(DIM_DENSE);
  EXPECT_TRUE(layout.dim_unique(0));

  layout.add_dim_level_type(DIM_COMPRESSED);
  EXPECT_TRUE(layout.dim_unique(1));
}

TEST(Layout, DimensionIsOrderedByDefault) {
  Layout layout({0, 1});
  layout.add_dim_level_type(DIM_DENSE);
  EXPECT_TRUE(layout.dim_ordered(0));

  layout.add_dim_level_type(DIM_COMPRESSED);
  EXPECT_TRUE(layout.dim_ordered(1));
}

TEST(Layout, DeleteDimensionWorksForDeletingLastDimFromDenseLayout) {
  Layout layout({0, 1});
  layout.add_dim_level_type(DIM_DENSE);
  layout.add_dim_level_type(DIM_DENSE);
  layout.add_dim_unique(false);
  layout.add_dim_unique(true);
  ASSERT_TRUE(LayoutUtil::IsDense(layout));
  ASSERT_EQ(layout.minor_to_major().size(), 2);
  ASSERT_EQ(layout.dim_unique_size(), 2);

  layout.DeleteDimension(1);
  EXPECT_THAT(layout.minor_to_major(), ElementsAre(0));
  ASSERT_EQ(layout.dim_level_types_size(), 1);
  EXPECT_EQ(layout.dim_level_type(0), DIM_DENSE);
  ASSERT_EQ(layout.dim_unique_size(), 1);
  EXPECT_FALSE(layout.dim_unique(0));
}

TEST(Layout, DeleteDimensionWorksForDeletingNonLastDimFromDenseLayout) {
  Layout layout({1, 0});
  layout.add_dim_level_type(DIM_DENSE);
  layout.add_dim_level_type(DIM_DENSE);
  layout.add_dim_unique(false);
  layout.add_dim_unique(true);
  ASSERT_TRUE(LayoutUtil::IsDense(layout));
  ASSERT_EQ(layout.minor_to_major().size(), 2);
  ASSERT_EQ(layout.dim_unique_size(), 2);

  layout.DeleteDimension(0);
  EXPECT_THAT(layout.minor_to_major(), ElementsAre(0));
  ASSERT_EQ(layout.dim_level_types_size(), 1);
  EXPECT_EQ(layout.dim_level_type(0), DIM_DENSE);
  ASSERT_EQ(layout.dim_unique_size(), 1);
  EXPECT_TRUE(layout.dim_unique(0));
}

TEST(Layout, DeleteDimensionWorksForDeletingLastDimFromSparseLayout) {
  Layout layout({0, 1});
  layout.add_dim_level_type(DIM_COMPRESSED);
  layout.add_dim_level_type(DIM_DENSE);
  layout.add_dim_unique(false);
  layout.add_dim_unique(true);
  ASSERT_TRUE(LayoutUtil::IsSparse(layout));
  ASSERT_EQ(layout.minor_to_major().size(), 2);
  ASSERT_EQ(layout.dim_unique_size(), 2);

  layout.DeleteDimension(1);
  EXPECT_THAT(layout.minor_to_major(), ElementsAre(0));
  ASSERT_EQ(layout.dim_level_types_size(), 1);
  EXPECT_EQ(layout.dim_level_type(0), DIM_COMPRESSED);
  ASSERT_EQ(layout.dim_unique_size(), 1);
  EXPECT_FALSE(layout.dim_unique(0));
}

TEST(Layout, DeleteDimensionWorksForDeletingNonLastDimFromSparseLayout) {
  Layout layout({1, 0});
  layout.add_dim_level_type(DIM_COMPRESSED);
  layout.add_dim_level_type(DIM_DENSE);
  layout.add_dim_unique(false);
  layout.add_dim_unique(true);
  ASSERT_TRUE(LayoutUtil::IsSparse(layout));
  ASSERT_EQ(layout.minor_to_major().size(), 2);
  ASSERT_EQ(layout.dim_unique_size(), 2);

  layout.DeleteDimension(0);
  EXPECT_THAT(layout.minor_to_major(), ElementsAre(0));
  ASSERT_EQ(layout.dim_level_types_size(), 1);
  EXPECT_EQ(layout.dim_level_type(0), DIM_DENSE);
  ASSERT_EQ(layout.dim_unique_size(), 1);
  EXPECT_TRUE(layout.dim_unique(0));
}

}  // namespace
}  // namespace xla
