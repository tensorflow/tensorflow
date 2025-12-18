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

TEST(NamedShardingTest, ToString) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});

  AxisRef axis_a(0);
  AxisRef axis_b(1, {2, 2});
  AxisRef axis_c(2);
  AxisRef axis_d(3, {4, 2});

  DimensionSharding ds_empty;
  EXPECT_EQ(ds_empty.ToString(mesh), "{}");

  DimensionSharding ds_empty_open(/*axes=*/{}, /*is_closed=*/false);
  EXPECT_EQ(ds_empty_open.ToString(mesh), "{?}");

  DimensionSharding ds_a({axis_a}, /*is_closed=*/true);
  EXPECT_EQ(ds_a.ToString(mesh), "{a}");

  DimensionSharding ds_ab({axis_a, axis_b}, /*is_closed=*/true);
  EXPECT_EQ(ds_ab.ToString(mesh), "{a, b:(2)2}");

  DimensionSharding ds_ab_open({axis_a, axis_b}, /*is_closed=*/false);
  EXPECT_EQ(ds_ab_open.ToString(mesh), "{a, b:(2)2, ?}");

  NamedSharding sharding_dim(mesh, {ds_a, ds_ab_open});
  EXPECT_EQ(sharding_dim.ToString(),
            "{@mesh<a=2,b=4,c=3,d=8>, [{a}, {a, b:(2)2, ?}]}");

  NamedSharding sharding_replicated(mesh, {}, {axis_c});
  EXPECT_EQ(sharding_replicated.ToString(),
            "{@mesh<a=2,b=4,c=3,d=8>, [], replicated={c}}");

  NamedSharding sharding_unreduced(mesh, {}, {}, {axis_d});
  EXPECT_EQ(sharding_unreduced.ToString(),
            "{@mesh<a=2,b=4,c=3,d=8>, [], unreduced={d:(4)2}}");

  Mesh maximal_mesh(5);
  NamedSharding maximal_sharding(maximal_mesh);
  EXPECT_EQ(maximal_sharding.ToString(), "{@maximal_mesh<device_ids=[5]>, []}");

  OpMetadata metadata;
  metadata.set_op_name("foo");
  NamedSharding sharding_all(mesh, {ds_a}, {axis_c}, {axis_d}, {metadata});
  EXPECT_EQ(
      sharding_all.ToString(),
      "{@mesh<a=2,b=4,c=3,d=8>, [{a}], replicated={c}, unreduced={d:(4)2}}");
  // EXPECT_EQ(
  //     sharding_all.ToString(/*include_metadata=*/true),
  //     "{@mesh<a=2,b=4,c=3,d=8>, dim_shardings={{a}}, replicated_axes={c}, "
  //     "unreduced_axes={d:(4)2}, metadata={op_name: \"foo\"}}");

  // TODO: Add tests for mesh with non iota device assignment.
}

TEST(NamedShardingTest, DimensionShardingSplit) {
  Mesh mesh({2, 4, 3, 8}, {"a", "b", "c", "d"});

  AxisRef a(0), b(1), c(2), d(3);
  AxisRef b1(1, {1, 2}), b2(1, {2, 2});
  AxisRef d1(3, {1, 4}), d2(3, {4, 2});

  DimensionSharding ds1({a, b1, c}, /*is_closed=*/true);
  DimensionSharding split1 = ds1.split(mesh, 6);
  EXPECT_THAT(split1.axes(), ElementsAre(a, c));
  EXPECT_THAT(ds1.axes(), ElementsAre(b1));

  DimensionSharding ds2({a, b, c}, /*is_closed=*/true);
  DimensionSharding split2 = ds2.split(mesh, 4);
  EXPECT_THAT(split2.axes(), ElementsAre(a, b1));
  EXPECT_THAT(ds2.axes(), ElementsAre(b2, c));
  DimensionSharding split3 = ds2.split(mesh, 2);
  EXPECT_THAT(split3.axes(), ElementsAre(b2));
  EXPECT_THAT(ds2.axes(), ElementsAre(c));

  DimensionSharding ds3({b, d}, /*is_closed=*/true);
  DimensionSharding split4 = ds3.split(mesh, 16);
  EXPECT_THAT(split4.axes(), ElementsAre(b, d1));
  EXPECT_THAT(ds3.axes(), ElementsAre(d2));

  DimensionSharding ds4({a, c}, /*is_closed=*/true);
  DimensionSharding split5 = ds4.split(mesh, 3);
  EXPECT_THAT(split5.axes(), ElementsAre(c));
  EXPECT_THAT(ds4.axes(), ElementsAre(a));

  // TODO: Should we merge axes while pushing in dimension sharding, it would
  // only be required if we allow non-sorted or non-merged axes in
  // dimension sharding construction.
  DimensionSharding ds5({b1, c, b2}, /*is_closed=*/true);
  DimensionSharding split6 = ds5.split(mesh, 3);
  EXPECT_THAT(split6.axes(), ElementsAre(c));
  EXPECT_THAT(ds5.axes(), ElementsAre(b1, b2));
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
