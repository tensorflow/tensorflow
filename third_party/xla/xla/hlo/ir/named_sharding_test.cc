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
