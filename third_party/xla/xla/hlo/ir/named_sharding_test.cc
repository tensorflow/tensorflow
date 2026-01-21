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
#include "xla/hlo/ir/tile_assignment.h"
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
  Mesh mesh_abcde({2, 4, 3, 8, 2}, {"a", "b", "c", "d", "e"});
  AxisRef axis_a(0);
  AxisRef axis_b(1);
  AxisRef axis_c(2);
  AxisRef axis_d(3);
  AxisRef axis_e(4);

  NamedSharding sharding =
      test_utils::FromAxisNames(mesh_abcde, /*dim_shardings=*/{{"c"}, {"b"}},
                                /*replicated_axes=*/{"a"},
                                /*unreduced_axes=*/{"d"},
                                /*manual_axes=*/{"e"});
  DimensionSharding ds_c({axis_c}, /*is_closed=*/true);
  DimensionSharding ds_b({axis_b}, /*is_closed=*/true);

  EXPECT_EQ(sharding, NamedSharding(mesh_abcde, {ds_c, ds_b}, {axis_a},
                                    {axis_d}, {axis_e}));

  NamedSharding sharding2 = test_utils::FromAxisNames(
      mesh_abcde,
      /*dim_shardings=*/{{"c", "a"}, {}, {"b"}},
      /*replicated_axes=*/{"d"}, /*unreduced_axes=*/{"e"});
  DimensionSharding ds_ca({axis_c, axis_a}, /*is_closed=*/true);

  EXPECT_EQ(sharding2,
            NamedSharding(mesh_abcde, {ds_ca, DimensionSharding(), ds_b},
                          {axis_d}, {axis_e}));
}

class NamedShardingEqualityTest : public ::testing::Test {
 protected:
  const Mesh mesh_abcde_ = Mesh({2, 4, 3, 8, 2}, {"a", "b", "c", "d", "e"});
  const NamedSharding base_ = test_utils::FromAxisNames(
      mesh_abcde_, /*dim_shardings=*/{{"a", "b:(2)2"}, {"d:(4)2", "c"}},
      /*replicated_axes=*/{"b:(2)2"},
      /*unreduced_axes=*/{"c"});
};

TEST_F(NamedShardingEqualityTest, BaseEquality) {
  EXPECT_EQ(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"b:(2)2"}, {"c"}));
}

TEST_F(NamedShardingEqualityTest, EqualEvenWithDifferentMeshAxisNames) {
  // Equal even with different mesh axis names
  Mesh mesh_cadbe({2, 4, 3, 8, 2}, {"c", "a", "d", "b", "e"});
  EXPECT_EQ(base_, test_utils::FromAxisNames(mesh_cadbe,
                                             {{"c", "a:(2)2"}, {"b:(4)2", "d"}},
                                             {"a:(2)2"}, {"d"}));
}

TEST_F(NamedShardingEqualityTest, EqualEvenWithDifferentMetadata) {
  // Equal even with different metadata.
  OpMetadata metadata;
  metadata.set_op_name("foo");
  EXPECT_EQ(base_, test_utils::FromAxisNames(
                       mesh_abcde_, {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                       {"b:(2)2"}, {"c"}, {}, {metadata}));
}

TEST_F(NamedShardingEqualityTest, DifferentDimShardings) {
  // Different dim_shardings
  EXPECT_NE(base_, test_utils::FromAxisNames(
                       mesh_abcde_, {{"a", "b:(2)2", "?"}, {"d:(4)2", "c"}},
                       {"b:(2)2"}, {"c"}));
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"d:(4)2", "c"}, {"a", "b:(2)2"}},
                                             {"b:(2)2"}, {"c"}));
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_, {{"a", "b:(2)2"}},
                                             {"b:(2)2"}, {"c"}));
}

TEST_F(NamedShardingEqualityTest, DifferentReplicatedAxes) {
  // Different replicated_axes
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"d:(4)2"}, {"c"}));
}

TEST_F(NamedShardingEqualityTest, DifferentUnreducedAxes) {
  // Different unreduced_axes
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"b:(2)2"}, {"a"}));
}

TEST_F(NamedShardingEqualityTest, DifferentManualAxes) {
  // Different manual_axes
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"b:(2)2"}, {"c"}, {"e"}));
}

TEST_F(NamedShardingEqualityTest, DifferentMeshShape) {
  // Different mesh shape
  Mesh mesh_diff_shape({2, 4, 3, 9, 2}, {"a", "b", "c", "d", "e"});
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_diff_shape,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"b:(2)2"}, {"c"}));
}

TEST(NamedShardingTest, ToString) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});

  AxisRef axis_a(0);
  AxisRef axis_b(1, {2, 2});
  AxisRef axis_c(2);
  AxisRef axis_d(3, {4, 2});

  DimensionSharding ds_empty;
  EXPECT_EQ(ds_empty.ToString(&mesh), "{}");

  DimensionSharding ds_empty_open(/*axes=*/{}, /*is_closed=*/false);
  EXPECT_EQ(ds_empty_open.ToString(&mesh), "{?}");

  DimensionSharding ds_a({axis_a}, /*is_closed=*/true);
  EXPECT_EQ(ds_a.ToString(&mesh), "{a}");

  DimensionSharding ds_ab({axis_a, axis_b}, /*is_closed=*/true);
  EXPECT_EQ(ds_ab.ToString(&mesh), "{a, b:(2)2}");

  DimensionSharding ds_ab_open({axis_a, axis_b}, /*is_closed=*/false);
  EXPECT_EQ(ds_ab_open.ToString(&mesh), "{a, b:(2)2, ?}");

  DimensionSharding ds_c({axis_c}, /*is_closed=*/true);
  NamedSharding sharding_dim(mesh, {ds_c, ds_ab_open});
  EXPECT_EQ(sharding_dim.ToString(),
            "{@mesh<a=2,b=4,c=3,d=8>, [{c}, {a, b:(2)2, ?}]}");

  NamedSharding sharding_fully_replicated(mesh);
  EXPECT_EQ(sharding_fully_replicated.ToString(), "{replicated}");

  NamedSharding sharding_replicated =
      test_utils::FromAxisNames(mesh, {}, {"c"});
  EXPECT_EQ(sharding_replicated.ToString(),
            "{@mesh<a=2,b=4,c=3,d=8>, [], replicated={c}}");

  NamedSharding sharding_unreduced =
      test_utils::FromAxisNames(mesh, {}, {}, {"d:(4)2"});
  EXPECT_EQ(sharding_unreduced.ToString(),
            "{@mesh<a=2,b=4,c=3,d=8>, [], unreduced={d:(4)2}}");

  Mesh maximal_mesh(5);
  NamedSharding maximal_sharding(maximal_mesh);
  EXPECT_EQ(maximal_sharding.ToString(), "{maximal device=5}");

  Mesh non_iota_mesh(
      TileAssignment(/*dims=*/{2, 4, 4, 2}, /*reshape_dims=*/{1, 4, 1, 16},
                     /*transpose_perm=*/{2, 3, 0, 1}),
      {"a", "b", "c", "d"});
  NamedSharding sharding_non_iota =
      test_utils::FromAxisNames(non_iota_mesh, {{"a"}});
  EXPECT_EQ(sharding_non_iota.ToString(),
            "{@mesh<a=2,b=4,c=4,d=2>, device_ids=([4,16]T(1,0)), [{a}]}");

  OpMetadata metadata1;
  metadata1.set_op_name("foo");
  OpMetadata metadata2;
  metadata2.set_op_name("bar");
  NamedSharding sharding_all = test_utils::FromAxisNames(
      mesh, {{"a"}}, {"c"}, {"d:(4)2"}, {"b:(2)2"}, {metadata1, metadata2});
  EXPECT_EQ(sharding_all.ToString(),
            "{@mesh<a=2,b=4,c=3,d=8>, [{a}], replicated={c}, "
            "unreduced={d:(4)2}, manual={b:(2)2}}");
  EXPECT_EQ(sharding_all.ToString(/*include_metadata=*/true),
            "{@mesh<a=2,b=4,c=3,d=8>, [{a}], replicated={c}, "
            "unreduced={d:(4)2}, manual={b:(2)2}, metadata={{op_name=\"foo\"}, "
            "{op_name=\"bar\"}}}");
}

TEST(NamedShardingTest, DimensionShardingAppend) {
  Mesh mesh{{2, 4, 8}, {"a", "b", "c"}};
  AxisRef a(0), b(1), c(2);
  AxisRef b1(1, {1, 2}), b2(1, {2, 2});

  {
    DimensionSharding ds1({}, /*is_closed=*/true);
    DimensionSharding ds2({}, /*is_closed=*/true);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre());
  }

  {
    DimensionSharding ds1({a}, /*is_closed=*/true);
    DimensionSharding ds2({}, /*is_closed=*/true);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre(a));

    DimensionSharding ds3({}, /*is_closed=*/false);
    DimensionSharding ds4({a}, /*is_closed=*/true);
    ds3.Append(ds4, mesh);
    EXPECT_THAT(ds3.axes(), ElementsAre(a));
    EXPECT_FALSE(ds3.is_closed());
  }

  {
    DimensionSharding ds1({a}, /*is_closed=*/true);
    DimensionSharding ds2({b}, /*is_closed=*/true);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre(a, b));
  }

  {
    DimensionSharding ds1({a}, /*is_closed=*/true);
    DimensionSharding ds2({b}, /*is_closed=*/false);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre(a, b));
    EXPECT_TRUE(ds1.is_closed());
  }

  {
    DimensionSharding ds1({a}, /*is_closed=*/false);
    DimensionSharding ds2({b}, /*is_closed=*/true);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre(a, b));
    EXPECT_FALSE(ds1.is_closed());
  }

  {
    DimensionSharding ds1({b1}, /*is_closed=*/true);
    DimensionSharding ds2({b2}, /*is_closed=*/true);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre(b));
  }

  {
    DimensionSharding ds1({a, b1}, /*is_closed=*/true);
    DimensionSharding ds2({b2, c}, /*is_closed=*/true);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre(a, b, c));
  }

  {
    DimensionSharding ds1({a, b1}, /*is_closed=*/true);
    DimensionSharding ds2({c, b2}, /*is_closed=*/true);
    ds1.Append(ds2, mesh);
    EXPECT_THAT(ds1.axes(), ElementsAre(a, b1, c, b2));
  }
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
