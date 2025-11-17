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

#include <gtest/gtest.h>
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using DimensionSharding = NamedSharding::DimensionSharding;

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

}  // namespace
}  // namespace xla
