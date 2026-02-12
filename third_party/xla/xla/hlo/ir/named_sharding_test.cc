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
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
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
      /*replicated_axes=*/{"e"},
      /*unreduced_axes=*/{"b:(1)2"});
};

TEST_F(NamedShardingEqualityTest, BaseEquality) {
  EXPECT_EQ(base_, test_utils::FromAxisNames(
                       mesh_abcde_,
                       /*dim_shardings=*/{{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                       /*replicated_axes=*/{"e"},
                       /*unreduced_axes=*/{"b:(1)2"}));
}

TEST_F(NamedShardingEqualityTest, EqualEvenWithDifferentMeshAxisNames) {
  Mesh mesh_cadbe({2, 4, 3, 8, 2}, {"c", "a", "d", "b", "e"});

  EXPECT_EQ(base_, test_utils::FromAxisNames(
                       mesh_cadbe,
                       /*dim_shardings=*/{{"c", "a:(2)2"}, {"b:(4)2", "d"}},
                       /*replicated_axes=*/{"e"},
                       /*unreduced_axes=*/{"a:(1)2"}));
}

TEST_F(NamedShardingEqualityTest, EqualEvenWithDifferentMetadata) {
  // Equal even with different metadata.
  OpMetadata metadata;
  metadata.set_op_name("foo");

  EXPECT_EQ(base_, test_utils::FromAxisNames(
                       mesh_abcde_,
                       /*dim_shardings=*/{{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                       /*replicated_axes=*/{"e"},
                       /*unreduced_axes=*/{"b:(1)2"}, {}, {metadata}));
}

TEST_F(NamedShardingEqualityTest, DifferentDimShardings) {
  EXPECT_NE(base_, test_utils::FromAxisNames(
                       mesh_abcde_, {{"a", "b:(2)2", "?"}, {"d:(4)2", "c"}},
                       {"e"}, {"b:(1)2"}));
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"d:(4)2", "c"}, {"a", "b:(2)2"}},
                                             {"e"}, {"b:(1)2"}));
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_, {{"a", "b:(2)2"}},
                                             {"e"}, {"b:(1)2"}));
}

TEST_F(NamedShardingEqualityTest, DifferentReplicatedAxes) {
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"d:(1)4"}, {"b:(1)2"}));
}

TEST_F(NamedShardingEqualityTest, DifferentUnreducedAxes) {
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"e"}, {"d:(1)4"}));
}

TEST_F(NamedShardingEqualityTest, DifferentManualAxes) {
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_abcde_,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"e"}, {"b:(1)2"}, {"d:(1)4"}));
}

TEST_F(NamedShardingEqualityTest, DifferentMeshShape) {
  Mesh mesh_diff_shape({2, 4, 3, 16, 2}, {"a", "b", "c", "d", "e"});
  EXPECT_NE(base_, test_utils::FromAxisNames(mesh_diff_shape,
                                             {{"a", "b:(2)2"}, {"d:(4)2", "c"}},
                                             {"e"}, {"b:(1)2"}));
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
            "{mesh[a=2,b=4,c=3,d=8], [{c}, {a, b:(2)2, ?}]}");

  NamedSharding sharding_fully_replicated = NamedSharding::Replicate();
  EXPECT_EQ(sharding_fully_replicated.ToString(), "{mesh[], replicated}");
  NamedSharding sharding_fully_replicated_with_mesh(mesh);
  EXPECT_EQ(sharding_fully_replicated_with_mesh.ToString(),
            "{mesh[a=2,b=4,c=3,d=8], replicated}");
  NamedSharding sharding_replicated =
      test_utils::FromAxisNames(mesh, {}, {"c"});
  EXPECT_EQ(sharding_replicated.ToString(),
            "{mesh[a=2,b=4,c=3,d=8], [], replicated={c}}");

  Mesh maximal_mesh(5);
  NamedSharding maximal_sharding(maximal_mesh);
  EXPECT_EQ(maximal_sharding.ToString(), "{maximal_mesh[device_id=5]}");

  NamedSharding sharding_fully_unreduced = NamedSharding::Unreduced(mesh);
  EXPECT_EQ(sharding_fully_unreduced.ToString(),
            "{mesh[a=2,b=4,c=3,d=8], unreduced}");
  NamedSharding sharding_unreduced =
      test_utils::FromAxisNames(mesh, {}, {}, {"d:(4)2"});
  EXPECT_EQ(sharding_unreduced.ToString(),
            "{mesh[a=2,b=4,c=3,d=8], [], unreduced={d:(4)2}}");

  NamedSharding sharding_fully_manual = NamedSharding::Manual(mesh);
  EXPECT_EQ(sharding_fully_manual.ToString(),
            "{mesh[a=2,b=4,c=3,d=8], manual}");
  NamedSharding sharding_manual =
      test_utils::FromAxisNames(mesh, {}, {}, {}, {"d:(4)2"});
  EXPECT_EQ(sharding_manual.ToString(),
            "{mesh[a=2,b=4,c=3,d=8], [], manual={d:(4)2}}");

  Mesh non_iota_mesh(
      TileAssignment(/*dims=*/{2, 4, 4, 2}, /*reshape_dims=*/{1, 4, 1, 16},
                     /*transpose_perm=*/{2, 3, 0, 1}),
      {"a", "b", "c", "d"});
  NamedSharding sharding_non_iota =
      test_utils::FromAxisNames(non_iota_mesh, {{"a"}});
  EXPECT_EQ(sharding_non_iota.ToString(),
            "{mesh[a=2,b=4,c=4,d=2], device_ids=([4,16]T(1,0)), [{a}]}");

  OpMetadata metadata1;
  metadata1.set_op_name("foo");
  OpMetadata metadata2;
  metadata2.set_op_name("bar");
  NamedSharding sharding_all = test_utils::FromAxisNames(
      mesh, {{"a"}}, {"c"}, {"d:(4)2"}, {"b:(2)2"}, {metadata1, metadata2});
  EXPECT_EQ(sharding_all.ToString(),
            "{mesh[a=2,b=4,c=3,d=8], [{a}], replicated={c}, "
            "unreduced={d:(4)2}, manual={b:(2)2}}");
  EXPECT_EQ(sharding_all.ToString(/*include_metadata=*/true),
            "{mesh[a=2,b=4,c=3,d=8], [{a}], replicated={c}, "
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

TEST(NamedShardingTest, IsPrefixOf) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});
  AxisRef a(0);
  AxisRef b(1);
  AxisRef c(2);
  AxisRef d(3);
  AxisRef b1(1, {1, 2});
  AxisRef b2(1, {2, 2});
  DimensionSharding ds_a({a}, /*is_closed=*/true);
  DimensionSharding ds_b({b}, /*is_closed=*/true);
  DimensionSharding ds_ab({a, b}, /*is_closed=*/true);
  DimensionSharding ds_ba({b, a}, /*is_closed=*/true);
  DimensionSharding ds_empty;
  DimensionSharding ds_b1({b1}, /*is_closed=*/true);
  DimensionSharding ds_b1_c({b1, c}, /*is_closed=*/true);

  // Identity
  EXPECT_TRUE(ds_a.IsPrefixOf(ds_a, mesh, mesh));
  // Empty is prefix of anything
  EXPECT_TRUE(ds_empty.IsPrefixOf(ds_a, mesh, mesh));
  EXPECT_TRUE(ds_empty.IsPrefixOf(ds_empty, mesh, mesh));
  // Proper prefix
  EXPECT_TRUE(ds_a.IsPrefixOf(ds_ab, mesh, mesh));
  // Not a prefix (order)
  EXPECT_FALSE(ds_a.IsPrefixOf(ds_ba, mesh, mesh));
  // Not a prefix (length)
  EXPECT_FALSE(ds_ab.IsPrefixOf(ds_a, mesh, mesh));
  // Not a prefix (mismatch)
  EXPECT_FALSE(ds_a.IsPrefixOf(ds_b, mesh, mesh));
  // b1 (2) is prefix of b (4)
  EXPECT_TRUE(ds_b1.IsPrefixOf(ds_b, mesh, mesh));
  // b (4) is NOT prefix of b1 (2)
  EXPECT_FALSE(ds_b.IsPrefixOf(ds_b1, mesh, mesh));
  // b (4) is NOT prefix of b1, c (because b is larger than b1)
  EXPECT_FALSE(ds_b.IsPrefixOf(ds_b1_c, mesh, mesh));
}

TEST(NamedShardingTest, IsPrefixOfDifferentMeshes) {
  Mesh mesh_a({2, 2, 4}, {"x", "y", "z"});
  Mesh mesh_b({4, 2, 2}, {"x", "y", "z"});
  AxisRef axis_1(1);
  DimensionSharding sharding_a({axis_1}, /*is_closed=*/true);
  DimensionSharding sharding_b({axis_1}, /*is_closed=*/true);

  EXPECT_FALSE(sharding_a.IsPrefixOf(sharding_b, mesh_a, mesh_b));
  EXPECT_FALSE(sharding_b.IsPrefixOf(sharding_a, mesh_b, mesh_a));
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

TEST(NamedShardingTest, ValidSharding) {
  Mesh mesh({2, 4}, {"a", "b"});
  NamedSharding sharding = test_utils::FromAxisNames(mesh, {{"a"}}, {"b"});
  TF_EXPECT_OK(VerifyNamedSharding(sharding));
}

TEST(NamedShardingTest, ValidShardingWithSubAxes) {
  Mesh mesh({4}, {"a"});
  NamedSharding sharding =
      test_utils::FromAxisNames(mesh, {{"a:(1)2"}}, {"a:(2)2"});
  TF_EXPECT_OK(VerifyNamedSharding(sharding));
}

TEST(NamedShardingTest, InvalidAxisIndex) {
  Mesh mesh({2}, {"a"});
  AxisRef b(1);  // Index 1 is out of bounds for size 1

  DimensionSharding ds_b({b}, /*is_closed=*/true);

  EXPECT_DEATH(NamedSharding(mesh, {ds_b}),
               "Axis index must be less than number of axes.*"
               "Axis index: 1, Number of axes: 1");
}

TEST(NamedShardingTest, OverlappingAxesSameDimValidation) {
  Mesh mesh({2}, {"a"});
  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a", "a"}}),
               "Axes cannot coexist or axes overlap");
}

TEST(NamedShardingTest, OverlappingAxesDifferentDimsValidation) {
  Mesh mesh({2}, {"a"});
  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a"}, {"a"}}),
               "Axes cannot coexist or axes overlap");
}

TEST(NamedShardingTest, OverlappingAxesDimAndReplicatedValidation) {
  Mesh mesh({2}, {"a"});
  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a"}}, {"a"}),
               "Axes cannot coexist or axes overlap");
}

TEST(NamedShardingTest, OverlappingAxesDimAndUnreducedValidation) {
  Mesh mesh({2}, {"a"});
  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a"}}, {}, {"a"}),
               "Axes cannot coexist or axes overlap");
}

TEST(NamedShardingTest, OverlappingAxesDimAndManualValidation) {
  Mesh mesh({2}, {"a"});
  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a"}}, {}, {}, {"a"}),
               "Axes cannot coexist or axes overlap");
}

TEST(NamedShardingTest, OverlappingSubAxesValidation) {
  Mesh mesh({4}, {"a"});

  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a:(1)4"}}, {"a:(1)2"}),
               "Sub-axis size must be strictly less than the full axis size.*"
               "Sub-axis size: 4, Axis size: 4");
}

TEST(NamedShardingTest, MergeableAxesValidation) {
  Mesh mesh({4}, {"a"});

  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a:(1)2", "a:(2)2"}}),
               "Adjacent axes in dimension sharding can be merged: "
               "a:\\(1\\)2, a:\\(2\\)2");
}

TEST(NamedShardingTest, SplitAxesValidation) {
  Mesh mesh({4}, {"a"});

  NamedSharding sharding =
      test_utils::FromAxisNames(mesh, {{"a:(2)2", "a:(1)2"}});
  TF_EXPECT_OK(VerifyNamedSharding(sharding));
}

TEST(NamedShardingTest, UnsortedReplicatedAxesValidation) {
  Mesh mesh({2, 2}, {"a", "b"});

  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {}, {"b", "a"}),
               "Replicated axes must be sorted by mesh axis index and "
               "sub-axis pre-size");
}

TEST(NamedShardingTest, UnsortedUnreducedAxesValidation) {
  Mesh mesh({2, 2}, {"a", "b"});

  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {}, {}, {"b", "a"}),
               "Unreduced axes must be sorted by mesh axis index and "
               "sub-axis pre-size");
}

TEST(NamedShardingTest, InvalidSubAxisDivisibility) {
  // 12 is divisible by 2 and 4, but not by 2*4=8.
  Mesh mesh({12}, {"a"});

  // pre_size=2, size=4.
  EXPECT_DEATH(test_utils::FromAxisNames(mesh, {{"a:(2)4"}}),
               "Sub-axis next_pre_size must divide the full axis size.*"
               "Next pre-size: 8, Axis size: 12");
}

TEST(NamedShardingPredicatesTest, IsReplicated) {
  Mesh mesh({2, 2}, {"a", "b"});
  NamedSharding sharding(mesh);
  EXPECT_TRUE(sharding.IsReplicated());
  EXPECT_FALSE(sharding.IsMaximal());
  EXPECT_FALSE(sharding.IsManual());
  EXPECT_FALSE(sharding.IsUnreduced());
}

TEST(NamedShardingPredicatesTest, IsMaximal) {
  Mesh mesh(1);
  NamedSharding sharding(mesh);
  EXPECT_TRUE(sharding.IsMaximal());
  EXPECT_FALSE(sharding.IsReplicated());
  EXPECT_FALSE(sharding.IsManual());
  EXPECT_FALSE(sharding.IsUnreduced());
}

TEST(NamedShardingPredicatesTest, IsUnreduced) {
  Mesh mesh({2, 2}, {"a", "b"});
  NamedSharding sharding1 = test_utils::FromAxisNames(mesh, {}, {}, {"a", "b"});
  EXPECT_TRUE(sharding1.IsUnreduced());
  EXPECT_FALSE(sharding1.IsReplicated());
  EXPECT_FALSE(sharding1.IsMaximal());
  EXPECT_FALSE(sharding1.IsManual());
}
TEST(NamedShardingPredicatesTest, IsUnreducedDoesntContainAllAxes) {
  Mesh mesh({2, 2}, {"a", "b"});
  NamedSharding sharding1 = test_utils::FromAxisNames(mesh, {}, {}, {"a"});
  EXPECT_FALSE(sharding1.IsUnreduced());
  EXPECT_FALSE(sharding1.IsReplicated());
  EXPECT_FALSE(sharding1.IsMaximal());
  EXPECT_FALSE(sharding1.IsManual());
}

TEST(NamedShardingPredicatesTest, IsManual) {
  Mesh mesh({2, 2}, {"a", "b"});
  NamedSharding sharding =
      test_utils::FromAxisNames(mesh, {}, {}, {}, {"a", "b"});
  EXPECT_TRUE(sharding.IsManual());
  EXPECT_FALSE(sharding.IsReplicated());
  EXPECT_FALSE(sharding.IsMaximal());
  EXPECT_FALSE(sharding.IsUnreduced());
}
TEST(NamedShardingPredicatesTest, IsManualDoesntContainAllAxes) {
  Mesh mesh({2, 2}, {"a", "b"});
  NamedSharding sharding = test_utils::FromAxisNames(mesh, {}, {}, {}, {"a"});
  EXPECT_FALSE(sharding.IsManual());
  EXPECT_FALSE(sharding.IsReplicated());
  EXPECT_FALSE(sharding.IsMaximal());
  EXPECT_FALSE(sharding.IsUnreduced());
}

TEST(NamedShardingTest, NamedShardingProtoConversion) {
  Mesh mesh({4, 4, 3, 5}, {"a", "b", "c", "d"});
  AxisRef axis_a_1(0, {1, 2});
  AxisRef axis_a_2(0, {2, 2});
  AxisRef axis_b(1);
  AxisRef axis_c(2);
  AxisRef axis_d(3);
  DimensionSharding ds_a1({axis_a_1}, /*is_closed=*/true);
  NamedSharding sharding(mesh, {ds_a1}, {axis_b}, {axis_d}, {axis_a_2, axis_c});

  NamedShardingProto proto = sharding.ToProto();

  ASSERT_THAT(
      proto,
      ::tsl::proto_testing::EquivToProto(
          ::tsl::proto_testing::ParseTextProtoOrDie<NamedShardingProto>(R"pb(
            mesh {
              axes { name: "a" size: 4 }
              axes { name: "b" size: 4 }
              axes { name: "c" size: 3 }
              axes { name: "d" size: 5 }
            }
            dim_shardings {
              axes {
                mesh_axis_index: 0
                sub_axis_info { pre_size: 1 size: 2 }
              }
              is_closed: true
            }
            replicated_axes { mesh_axis_index: 1 }
            unreduced_axes { mesh_axis_index: 3 }
            manual_axes {
              mesh_axis_index: 0
              sub_axis_info { pre_size: 2 size: 2 }
            }
            manual_axes { mesh_axis_index: 2 }
          )pb")));

  NamedSharding from_proto = NamedSharding::FromProto(proto);

  EXPECT_EQ(sharding, from_proto);
}

TEST(NamedShardingPredicatesTest, HasPartialReplication_Maximal) {
  EXPECT_FALSE(NamedSharding(Mesh(5)).HasPartialReplication());
}

TEST(NamedShardingPredicatesTest, HasPartialReplication_FullyReplicated) {
  EXPECT_FALSE(NamedSharding::Replicate().HasPartialReplication());
}

TEST(NamedShardingPredicatesTest,
     HasPartialReplication_FullyReplicatedWithMesh) {
  Mesh mesh({2, 2}, {"a", "b"});
  EXPECT_FALSE(NamedSharding(mesh).HasPartialReplication());
}

TEST(NamedShardingPredicatesTest, HasPartialReplication_ShardedOnAllAxes) {
  Mesh mesh({2, 2}, {"a", "b"});
  EXPECT_FALSE(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}, {"b"}})
                   .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest,
     HasPartialReplication_ShardedOnOneImplicitOnOther) {
  Mesh mesh({2, 2}, {"a", "b"});
  EXPECT_TRUE(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}})
                  .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest, HasPartialReplication_ExplicitReplicated) {
  Mesh mesh({2, 2}, {"a", "b"});
  EXPECT_TRUE(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}},
                                        /*replicated_axes=*/{"b"})
                  .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest,
     HasPartialReplication_UnreducedCoveringRemaining) {
  Mesh mesh({2, 2}, {"a", "b"});
  EXPECT_FALSE(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}},
                                         /*replicated_axes=*/{},
                                         /*unreduced_axes=*/{"b"})
                   .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest,
     HasPartialReplication_ManualCoveringRemaining) {
  Mesh mesh({2, 2}, {"a", "b"});
  EXPECT_FALSE(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}},
                                         /*replicated_axes=*/{},
                                         /*unreduced_axes=*/{},
                                         /*manual_axes=*/{"b"})
                   .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest,
     HasPartialReplication_MixedExplicitAndSharded) {
  Mesh mesh({2, 2}, {"a", "b"});
  EXPECT_TRUE(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}},
                                        /*replicated_axes=*/{"b"})
                  .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest,
     HasPartialReplication_ImplicitReplicationWithSubAxes) {
  // Mesh a=4. Sharding a:(2)2. Replicated on a:(1)2.
  EXPECT_TRUE(test_utils::FromAxisNames(Mesh({4}, {"a"}),
                                        /*dim_shardings=*/{{"a:(2)2"}})
                  .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest,
     HasPartialReplication_ExplicitReplicationWithSubAxes) {
  EXPECT_TRUE(test_utils::FromAxisNames(Mesh({4}, {"a"}),
                                        /*dim_shardings=*/{{"a:(2)2"}},
                                        /*replicated_axes=*/{"a:(1)2"})
                  .HasPartialReplication());
}

TEST(NamedShardingPredicatesTest, HasPartialReplication_UnreducedWithSubAxes) {
  EXPECT_FALSE(test_utils::FromAxisNames(Mesh({4}, {"a"}),
                                         /*dim_shardings=*/{{"a:(2)2"}},
                                         /*replicated_axes=*/{},
                                         /*unreduced_axes=*/{"a:(1)2"})
                   .HasPartialReplication());
}

}  // namespace
}  // namespace xla
