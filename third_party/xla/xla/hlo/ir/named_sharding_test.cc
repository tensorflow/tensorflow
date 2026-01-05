/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/ir/named_sharding.h"

#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using DimensionSharding = NamedSharding::DimensionSharding;
using ::testing::ElementsAre;

TEST(NamedShardingTest, CanonicalizedDimShardings) {
  Mesh mesh_abcd({2, 4}, {"a", "b"});

  DimensionSharding empty_ds;
  NamedSharding sharding1(mesh_abcd, {empty_ds, empty_ds});
  EXPECT_TRUE(sharding1.dim_shardings().empty());

  DimensionSharding ds_a({AxisRef(0)}, /*is_closed=*/true);
  NamedSharding sharding2(mesh_abcd, {ds_a, empty_ds});
  EXPECT_FALSE(sharding2.dim_shardings().empty());
}

TEST(NamedShardingTest, AxisNameCtor) {
  Mesh mesh_abcd({2, 4, 3, 8}, {"a", "b", "c", "d"});
  AxisRef axis_a(0);
  AxisRef axis_b(1);
  AxisRef axis_c(2);
  AxisRef axis_d(3);

  NamedSharding sharding =
      test_utils::FromAxisNames(mesh_abcd, /*dim_shardings=*/{{"c"}, {"b"}},
                                /*replicated_axes=*/{"a"},
                                /*unreduced_axes=*/{"d"});
  DimensionSharding ds_c({axis_c}, /*is_closed=*/true);
  DimensionSharding ds_b({axis_b}, /*is_closed=*/true);
  EXPECT_EQ(sharding,
            NamedSharding(mesh_abcd, {ds_c, ds_b}, {axis_a}, {axis_d}));

  NamedSharding sharding2 = test_utils::FromAxisNames(
      mesh_abcd,
      /*dim_shardings=*/{{"c", "a"}, {}, {"b"}},
      /*replicated_axes=*/{"d"}, /*unreduced_axes=*/{});
  DimensionSharding ds_ca({axis_c, axis_a}, /*is_closed=*/true);
  EXPECT_EQ(sharding2,
            NamedSharding(mesh_abcd, {ds_ca, DimensionSharding(), ds_b},
                          {axis_d}, {}));
}

TEST(NamedShardingTest, Equality) {
  Mesh mesh_abcd({2, 4, 3, 8}, {"a", "b", "c", "d"});

  AxisRef axis_a(0);
  AxisRef axis_b(1, {2, 2});
  AxisRef axis_c(2);
  AxisRef axis_d(3, {4, 2});

  DimensionSharding ds_ab({axis_a, axis_b}, /*is_closed=*/true);
  DimensionSharding ds_ab_open({axis_a, axis_b}, /*is_closed=*/false);
  DimensionSharding ds_dc({axis_d, axis_c}, /*is_closed=*/true);

  NamedSharding base(mesh_abcd, /*dim_shardings=*/{ds_ab, ds_dc},
                     /*replicated_axes=*/{axis_b},
                     /*unreduced_axes=*/{axis_c});

  EXPECT_EQ(base, NamedSharding(mesh_abcd, {ds_ab, ds_dc}, {axis_b}, {axis_c}));

  // Equal even with different mesh axis names
  Mesh mesh_cadb({2, 4, 3, 8}, {"c", "a", "d", "b"});
  EXPECT_EQ(base,
            NamedSharding(mesh_cadb, {ds_ab, ds_dc}, {axis_b}, {axis_c}, {}));

  // Equal even with different metadata.
  OpMetadata metadata;
  metadata.set_op_name("foo");
  EXPECT_EQ(base, NamedSharding(mesh_abcd, {ds_ab, ds_dc}, {axis_b}, {axis_c},
                                {metadata}));

  // Different dim_shardings
  EXPECT_NE(base,
            NamedSharding(mesh_abcd, {ds_ab_open, ds_dc}, {axis_b}, {axis_c}));
  EXPECT_NE(base, NamedSharding(mesh_abcd, {ds_dc, ds_ab}, {axis_b}, {axis_c}));
  EXPECT_NE(base, NamedSharding(mesh_abcd, {ds_ab}, {axis_b}, {axis_c}));

  // Different replicated_axes
  EXPECT_NE(base, NamedSharding(mesh_abcd, {ds_ab, ds_dc}, {axis_d}, {axis_c}));

  // Different unreduced_axes
  EXPECT_NE(base, NamedSharding(mesh_abcd, {ds_ab, ds_dc}, {axis_b}, {axis_a}));

  // Different mesh shape
  Mesh mesh_diff_shape({2, 4, 3, 9}, {"a", "b", "c", "d"});
  EXPECT_NE(base,
            NamedSharding(mesh_diff_shape, {ds_ab, ds_dc}, {axis_b}, {axis_c}));
}

class DimensionShardingSliceTest : public ::testing::Test {
 protected:
  Mesh mesh_{{2, 4, 3, 8, 1}, {"a", "b", "c", "d", "e"}};
  AxisRef a_{0}, b_{1}, c_{2}, d_{3}, e_{4};
  AxisRef b1_{1, {1, 2}}, b2_{1, {2, 2}};
  AxisRef d1_{3, {1, 4}}, d2_{3, {4, 2}};
};

TEST_F(DimensionShardingSliceTest, SliceMajorAxis) {
  DimensionSharding ds({a_, b_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 4);
  ASSERT_TRUE(slice.has_value());
  EXPECT_THAT(slice->axes(), ElementsAre(a_, b1_));
  EXPECT_THAT(ds.axes(), ElementsAre(b2_));
}

TEST_F(DimensionShardingSliceTest, SliceEntireAxis) {
  DimensionSharding ds({a_, b_, c_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 24);
  ASSERT_TRUE(slice.has_value());
  EXPECT_THAT(slice->axes(), ElementsAre(a_, b_, c_));
  EXPECT_THAT(ds.axes(), ElementsAre());
}

TEST_F(DimensionShardingSliceTest, SliceByOne) {
  DimensionSharding ds({a_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 1);
  ASSERT_TRUE(slice.has_value());
  EXPECT_THAT(slice->axes(), ElementsAre());
  EXPECT_THAT(ds.axes(), ElementsAre(a_));
}

TEST_F(DimensionShardingSliceTest, SliceSubAxis) {
  DimensionSharding ds({d1_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 2);
  ASSERT_TRUE(slice.has_value());
  EXPECT_THAT(slice->axes(), ElementsAre(AxisRef(3, {1, 2})));
  EXPECT_THAT(ds.axes(), ElementsAre(AxisRef(3, {2, 2})));
}

TEST_F(DimensionShardingSliceTest, SliceFurtherAxisOfSize1) {
  DimensionSharding ds({a_, e_, b_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 4);
  ASSERT_TRUE(slice.has_value());
  EXPECT_THAT(slice->axes(), ElementsAre(a_, e_, b1_));
  EXPECT_THAT(ds.axes(), ElementsAre(b2_));
}

TEST_F(DimensionShardingSliceTest, SliceFailsIfSizeNotDivisible) {
  DimensionSharding ds({a_, b_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 3);
  EXPECT_FALSE(slice.has_value());
}

TEST_F(DimensionShardingSliceTest, SliceFailsIfGcdIsOne) {
  DimensionSharding ds({a_, c_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 3);
  EXPECT_FALSE(slice.has_value());
}

TEST_F(DimensionShardingSliceTest, SliceFailsIfGcdIsOneForSecondAxis) {
  DimensionSharding ds({a_, b1_, c_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 6);
  EXPECT_FALSE(slice.has_value());
}

TEST_F(DimensionShardingSliceTest, SliceFailsIfAxisNotFullyDivisible) {
  DimensionSharding ds({b_, c_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 6);
  EXPECT_FALSE(slice.has_value());
}

TEST_F(DimensionShardingSliceTest, SequentialSlices) {
  DimensionSharding ds({a_, b_, c_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice1 = ds.Slice(mesh_, 4);
  ASSERT_TRUE(slice1.has_value());
  EXPECT_THAT(slice1->axes(), ElementsAre(a_, b1_));
  EXPECT_THAT(ds.axes(), ElementsAre(b2_, c_));
  std::optional<DimensionSharding> slice2 = ds.Slice(mesh_, 2);
  ASSERT_TRUE(slice2.has_value());
  EXPECT_THAT(slice2->axes(), ElementsAre(b2_));
  EXPECT_THAT(ds.axes(), ElementsAre(c_));
}

TEST_F(DimensionShardingSliceTest, SliceMajorAndSubAxis) {
  DimensionSharding ds({b_, d_}, /*is_closed=*/true);
  std::optional<DimensionSharding> slice = ds.Slice(mesh_, 16);
  ASSERT_TRUE(slice.has_value());
  EXPECT_THAT(slice->axes(), ElementsAre(b_, d1_));
  EXPECT_THAT(ds.axes(), ElementsAre(d2_));
}

TEST(NamedShardingTest, GetShardedSize) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});

  AxisRef axis_a(0);
  AxisRef axis_b(1, {2, 2});
  AxisRef axis_c(2);
  AxisRef axis_d(3, {4, 2});

  DimensionSharding ds_ab({axis_a, axis_b}, /*is_closed=*/true);
  EXPECT_EQ(ds_ab.getShardedSize(mesh), 2 * 2);

  DimensionSharding ds_dc({axis_d, axis_c}, /*is_closed=*/true);
  EXPECT_EQ(ds_dc.getShardedSize(mesh), 2 * 3);

  DimensionSharding ds_b({axis_b}, /*is_closed=*/true);
  EXPECT_EQ(ds_b.getShardedSize(mesh), 2);

  DimensionSharding ds_empty({}, /*is_closed=*/true);
  EXPECT_EQ(ds_empty.getShardedSize(mesh), 1);
}

TEST(NamedShardingTest, Dimension) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});

  AxisRef axis_a(0);
  AxisRef axis_b(1, {2, 2});
  AxisRef axis_c(2);
  AxisRef axis_d(3, {4, 2});

  DimensionSharding ds_ab({axis_a, axis_b}, /*is_closed=*/true);
  DimensionSharding ds_dc({axis_d, axis_c}, /*is_closed=*/true);

  NamedSharding sharding(mesh, /*dim_shardings=*/{ds_ab, ds_dc});

  EXPECT_EQ(sharding.dimension(0), 2 * 2);
  EXPECT_EQ(sharding.dimension(1), 2 * 3);
  EXPECT_EQ(sharding.num_dimensions(), 2);

  NamedSharding empty_sharding(mesh, /*dim_shardings=*/{});
  EXPECT_EQ(empty_sharding.num_dimensions(), 0);
}

TEST(NamedShardingTest, Dimensions) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});

  AxisRef axis_a(0);
  AxisRef axis_b(1, {2, 2});
  AxisRef axis_c(2);
  AxisRef axis_d(3, {4, 2});

  DimensionSharding ds_ab({axis_a, axis_b}, /*is_closed=*/true);
  DimensionSharding ds_dc({axis_d, axis_c}, /*is_closed=*/true);

  NamedSharding sharding(mesh, /*dim_shardings=*/{ds_ab, ds_dc});
  EXPECT_THAT(sharding.dimensions(), ElementsAre(2 * 2, 2 * 3));

  NamedSharding empty_sharding(mesh, /*dim_shardings=*/{});
  EXPECT_THAT(empty_sharding.dimensions(), ElementsAre());
}

TEST(NamedShardingTest, NumDevices) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});
  NamedSharding sharding(mesh, {});
  EXPECT_EQ(sharding.num_devices(), 2 * 4 * 3 * 8);

  Mesh maximal_mesh(5);
  NamedSharding maximal_sharding(maximal_mesh);
  EXPECT_EQ(maximal_sharding.num_devices(), 1);

  Mesh empty_mesh;
  NamedSharding empty_sharding(empty_mesh);
  EXPECT_EQ(empty_sharding.num_devices(), 0);
}

}  // namespace
}  // namespace xla
