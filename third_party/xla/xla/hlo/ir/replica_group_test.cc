/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/ir/replica_group.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/service/hlo.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {

CollectiveDeviceListProto CreateDeviceListProto(
    const std::vector<std::vector<int64_t>>& replica_groups) {
  CollectiveDeviceListProto proto;
  for (const auto& group : replica_groups) {
    auto* replica_group = proto.add_replica_groups();
    for (const auto& replica : group) {
      replica_group->add_replica_ids(replica);
    }
  }
  return proto;
}

TEST(MeshAxesReplicaGroupListTest, MaterializedReplicaGroups) {
  Mesh mesh_xy({2, 2}, {"x", "y"});

  EXPECT_DEATH(
      { MeshAxesReplicaGroupList replica_group_none(mesh_xy, {}); },
      "has only one device per replica group");

  MeshAxesReplicaGroupList replica_group_x(mesh_xy, {AxisRef(0)});
  std::vector<std::vector<int64_t>> expected_replica_groups_x = {{0, 2},
                                                                 {1, 3}};
  EXPECT_EQ(replica_group_x.flattened_replica_groups(),
            expected_replica_groups_x);

  MeshAxesReplicaGroupList replica_group_y(mesh_xy, {AxisRef(1)});
  std::vector<std::vector<int64_t>> expected_replica_groups_y = {{0, 1},
                                                                 {2, 3}};
  EXPECT_EQ(replica_group_y.flattened_replica_groups(),
            expected_replica_groups_y);

  MeshAxesReplicaGroupList replica_group_xy(mesh_xy, {AxisRef(0), AxisRef(1)});
  std::vector<std::vector<int64_t>> expected_replica_groups_xy = {{0, 1, 2, 3}};
  EXPECT_EQ(replica_group_xy.flattened_replica_groups(),
            expected_replica_groups_xy);
}

TEST(MeshAxesReplicaGroupListTest, MaterializedReplicaGroupsWithSubaxes) {
  Mesh mesh({6, 6}, {"a", "b"});

  // a:(1)2
  MeshAxesReplicaGroupList replica_group_a_1_2(mesh, {AxisRef(0, {1, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_2 = {
      {0, 18},  {1, 19},  {2, 20},  {3, 21},  {4, 22},  {5, 23},
      {6, 24},  {7, 25},  {8, 26},  {9, 27},  {10, 28}, {11, 29},
      {12, 30}, {13, 31}, {14, 32}, {15, 33}, {16, 34}, {17, 35}};
  EXPECT_EQ(replica_group_a_1_2.flattened_replica_groups(),
            expected_replica_groups_a_1_2);

  // a:(1)3
  MeshAxesReplicaGroupList replica_group_a_1_3(mesh, {AxisRef(0, {1, 3})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_3 = {
      {0, 12, 24}, {1, 13, 25}, {2, 14, 26},  {3, 15, 27},
      {4, 16, 28}, {5, 17, 29}, {6, 18, 30},  {7, 19, 31},
      {8, 20, 32}, {9, 21, 33}, {10, 22, 34}, {11, 23, 35}};
  EXPECT_EQ(replica_group_a_1_3.flattened_replica_groups(),
            expected_replica_groups_a_1_3);

  // a:(3)2
  MeshAxesReplicaGroupList replica_group_a_3_2(mesh, {AxisRef(0, {3, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_3_2 = {
      {0, 6},   {1, 7},   {2, 8},   {3, 9},   {4, 10},  {5, 11},
      {12, 18}, {13, 19}, {14, 20}, {15, 21}, {16, 22}, {17, 23},
      {24, 30}, {25, 31}, {26, 32}, {27, 33}, {28, 34}, {29, 35}};
  EXPECT_EQ(replica_group_a_3_2.flattened_replica_groups(),
            expected_replica_groups_a_3_2);

  // b:(1)2
  MeshAxesReplicaGroupList replica_group_b_1_2(mesh, {AxisRef(1, {1, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_b_1_2 = {
      {0, 3},   {1, 4},   {2, 5},   {6, 9},   {7, 10},  {8, 11},
      {12, 15}, {13, 16}, {14, 17}, {18, 21}, {19, 22}, {20, 23},
      {24, 27}, {25, 28}, {26, 29}, {30, 33}, {31, 34}, {32, 35}};
  EXPECT_EQ(replica_group_b_1_2.flattened_replica_groups(),
            expected_replica_groups_b_1_2);

  // b:(1)3
  MeshAxesReplicaGroupList replica_group_b_1_3(mesh, {AxisRef(1, {1, 3})});
  std::vector<std::vector<int64_t>> expected_replica_groups_b_1_3 = {
      {0, 2, 4},    {1, 3, 5},    {6, 8, 10},   {7, 9, 11},
      {12, 14, 16}, {13, 15, 17}, {18, 20, 22}, {19, 21, 23},
      {24, 26, 28}, {25, 27, 29}, {30, 32, 34}, {31, 33, 35}};
  EXPECT_EQ(replica_group_b_1_3.flattened_replica_groups(),
            expected_replica_groups_b_1_3);

  // b:(3)2
  MeshAxesReplicaGroupList replica_group_b_3_2(mesh, {AxisRef(1, {3, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_b_3_2 = {
      {0, 1},   {2, 3},   {4, 5},   {6, 7},   {8, 9},   {10, 11},
      {12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23},
      {24, 25}, {26, 27}, {28, 29}, {30, 31}, {32, 33}, {34, 35}};
  EXPECT_EQ(replica_group_b_3_2.flattened_replica_groups(),
            expected_replica_groups_b_3_2);

  // a:(1)2, b:(1)2
  MeshAxesReplicaGroupList replica_group_a_1_2_b_1_2(
      mesh, {AxisRef(0, {1, 2}), AxisRef(1, {1, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_2_b_1_2 = {
      {0, 3, 18, 21},   {1, 4, 19, 22},   {2, 5, 20, 23},
      {6, 9, 24, 27},   {7, 10, 25, 28},  {8, 11, 26, 29},
      {12, 15, 30, 33}, {13, 16, 31, 34}, {14, 17, 32, 35}};
  EXPECT_EQ(replica_group_a_1_2_b_1_2.flattened_replica_groups(),
            expected_replica_groups_a_1_2_b_1_2);

  // a:(1)3, b:(1)3
  MeshAxesReplicaGroupList replica_group_a_1_3_b_1_3(
      mesh, {AxisRef(0, {1, 3}), AxisRef(1, {1, 3})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_3_b_1_3 = {
      {0, 2, 4, 12, 14, 16, 24, 26, 28},
      {1, 3, 5, 13, 15, 17, 25, 27, 29},
      {6, 8, 10, 18, 20, 22, 30, 32, 34},
      {7, 9, 11, 19, 21, 23, 31, 33, 35}};
  EXPECT_EQ(replica_group_a_1_3_b_1_3.flattened_replica_groups(),
            expected_replica_groups_a_1_3_b_1_3);

  //  b:(1)3, a:(1)3 (Reverse order from above). This should produce the same
  // replica groups as the above but with ids in a different order.
  MeshAxesReplicaGroupList replica_group_b_1_3_a_1_3(
      mesh, {AxisRef(1, {1, 3}), AxisRef(0, {1, 3})});
  std::vector<std::vector<int64_t>> expected_replica_groups_b_1_3_a_1_3 = {
      {0, 12, 24, 2, 14, 26, 4, 16, 28},
      {1, 13, 25, 3, 15, 27, 5, 17, 29},
      {6, 18, 30, 8, 20, 32, 10, 22, 34},
      {7, 19, 31, 9, 21, 33, 11, 23, 35}};
  EXPECT_EQ(replica_group_a_1_3_b_1_3.flattened_replica_groups(),
            expected_replica_groups_a_1_3_b_1_3);

  // a:(3)2, b:(3)2
  MeshAxesReplicaGroupList replica_group_a_3_2_b_3_2(
      mesh, {AxisRef(0, {3, 2}), AxisRef(1, {3, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_3_2_b_3_2 = {
      {0, 1, 6, 7},     {2, 3, 8, 9},     {4, 5, 10, 11},
      {12, 13, 18, 19}, {14, 15, 20, 21}, {16, 17, 22, 23},
      {24, 25, 30, 31}, {26, 27, 32, 33}, {28, 29, 34, 35}};
  EXPECT_EQ(replica_group_a_3_2_b_3_2.flattened_replica_groups(),
            expected_replica_groups_a_3_2_b_3_2);
}

TEST(MeshAxesReplicaGroupListTest, MaterializedReplicaGroupsMatchExpectedV2) {
  Mesh mesh({8}, {"a"});

  // a:(1)2 -> replica_groups=[4,2]<=[2,4]T(1,0)
  MeshAxesReplicaGroupList v3_subaxis_1_2(mesh, {AxisRef(0, {1, 2})});
  IotaReplicaGroupList v2_subaxis_1_2(4, 2, {2, 4}, {1, 0});
  EXPECT_EQ(v3_subaxis_1_2.flattened_replica_groups(),
            v2_subaxis_1_2.flattened_replica_groups());

  // a:(1)4 -> replica_groups=[2,4]<=[4,2]T(1,0)
  MeshAxesReplicaGroupList v3_subaxis_1_4(mesh, {AxisRef(0, {1, 4})});
  IotaReplicaGroupList v2_subaxis_1_4(2, 4, {4, 2}, {1, 0});
  EXPECT_EQ(v3_subaxis_1_4.flattened_replica_groups(),
            v2_subaxis_1_4.flattened_replica_groups());

  // a:(2)2 -> replica_groups=[4,2]<=[2,2,2]T(0,2,1)
  MeshAxesReplicaGroupList v3_subaxis_2_2(mesh, {AxisRef(0, {2, 2})});
  IotaReplicaGroupList v2_subaxis_2_2(4, 2, {2, 2, 2}, {0, 2, 1});
  EXPECT_EQ(v3_subaxis_2_2.flattened_replica_groups(),
            v2_subaxis_2_2.flattened_replica_groups());

  // a:(2)4 -> replica_groups=[2,4]<=[8]
  MeshAxesReplicaGroupList v3_subaxis_2_4(mesh, {AxisRef(0, {2, 4})});
  IotaReplicaGroupList v2_subaxis_2_4(2, 4, {8}, {0});
  EXPECT_EQ(v3_subaxis_2_4.flattened_replica_groups(),
            v2_subaxis_2_4.flattened_replica_groups());

  // a:(4)2 -> replica_groups=[4,2]<=[8]
  MeshAxesReplicaGroupList v3_subaxis_4_2(mesh, {AxisRef(0, {4, 2})});
  IotaReplicaGroupList v2_subaxis_4_2(4, 2, {8}, {0});
  EXPECT_EQ(v3_subaxis_4_2.flattened_replica_groups(),
            v2_subaxis_4_2.flattened_replica_groups());

  //  {a:(1)2, a:(4)2} -> replica_groups=[2,4]<=[2,2,2]T(1,0,2)
  MeshAxesReplicaGroupList v3_subaxis_1_2_and_4_2(
      mesh, {AxisRef(0, {1, 2}), AxisRef(0, {4, 2})});
  IotaReplicaGroupList v2_subaxis_1_2_and_4_2(2, 4, {2, 2, 2}, {1, 0, 2});
  EXPECT_EQ(v3_subaxis_1_2_and_4_2.flattened_replica_groups(),
            v2_subaxis_1_2_and_4_2.flattened_replica_groups());

  //  {a:(4)2, a:(1)2} -> replica_groups=[2,4]<=[2,2,2]T(1,2,0)
  MeshAxesReplicaGroupList v3_subaxis_4_2_and_1_2(
      mesh, {AxisRef(0, {4, 2}), AxisRef(0, {1, 2})});
  IotaReplicaGroupList v2_subaxis_4_2_and_1_2(2, 4, {2, 2, 2}, {1, 2, 0});
  EXPECT_EQ(v3_subaxis_4_2_and_1_2.flattened_replica_groups(),
            v2_subaxis_4_2_and_1_2.flattened_replica_groups());

  // a      -> replica_groups=[1,8]<=[8]
  MeshAxesReplicaGroupList v3_no_subaxis(mesh, {AxisRef(0)});
  IotaReplicaGroupList v2_no_subaxis(1, 8, {8}, {0});
  EXPECT_EQ(v3_no_subaxis.flattened_replica_groups(),
            v2_no_subaxis.flattened_replica_groups());
}

TEST(MeshAxesReplicaGroupListTest,
     MaterializedReplicaGroupsRespectNonIotaDeviceOrdering) {
  // Create a mesh with non-iota device ordering.
  Array2D<int64_t> array({{3, 1}, {0, 2}});
  TileAssignment tile_assignment(std::make_shared<Array<int64_t>>(array));
  Mesh mesh_xy(tile_assignment, {"x", "y"});

  // Reduce along x axis.
  MeshAxesReplicaGroupList replica_group_x(mesh_xy, {AxisRef(0)});
  // With iota device ordering, the expected replica groups would be
  // {{0, 2}, {1, 3}}.
  std::vector<std::vector<int64_t>> expected_replica_groups_x = {{3, 0},
                                                                 {1, 2}};
  EXPECT_THAT(replica_group_x.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_x));

  // Reduce along y axis.
  MeshAxesReplicaGroupList replica_group_y(mesh_xy, {AxisRef(1)});
  // With iota device ordering, the expected replica groups would be
  // {{0, 1}, {2, 3}}.
  std::vector<std::vector<int64_t>> expected_replica_groups_y = {{3, 1},
                                                                 {0, 2}};
  EXPECT_THAT(replica_group_y.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_y));
}

TEST(MeshAxesReplicaGroupListTest, NumReplicaGroups) {
  Mesh all_axes({4, 4}, {"x", "y"});
  MeshAxesReplicaGroupList replica_group_across_all_axes(
      all_axes, {AxisRef(0), AxisRef(1)});
  EXPECT_EQ(replica_group_across_all_axes.num_replica_groups(), 1);
  EXPECT_EQ(replica_group_across_all_axes.num_devices_per_group(), 16);

  Mesh one_axes({3, 5}, {"a", "b"});
  MeshAxesReplicaGroupList replica_group_across_a(one_axes, {AxisRef(0)});
  MeshAxesReplicaGroupList replica_group_across_b(one_axes, {AxisRef(1)});
  EXPECT_EQ(replica_group_across_a.num_replica_groups(), 5);
  EXPECT_EQ(replica_group_across_a.num_devices_per_group(), 3);
  EXPECT_EQ(replica_group_across_b.num_replica_groups(), 3);
  EXPECT_EQ(replica_group_across_b.num_devices_per_group(), 5);
}

TEST(MeshAxesReplicaGroupListTest, ValidateSubAxesCoexistenceCheck) {
  Mesh mesh({8}, {"a"});
  MeshAxesReplicaGroupList replica_group_multiple_subaxes1(
      mesh, {AxisRef(0, {1, 2}), AxisRef(0, {4, 2})});
  MeshAxesReplicaGroupList replica_group_multiple_subaxes2(
      mesh, {AxisRef(0, {4, 2}), AxisRef(0, {1, 2})});

  Mesh overlap_mesh({2 * 3 * 5}, {"u"});
  EXPECT_DEATH(
      {
        MeshAxesReplicaGroupList overlapping_subaxes(
            overlap_mesh, {AxisRef(0, {6, 5}), AxisRef(0, {10, 3})});
      },
      "Axes cannot coexist or axes overlap.");
}

TEST(MeshAxesReplicaGroupListTest, ReplicaGroupsCountAndSizeForSubaxes) {
  Mesh mesh_one_subaxis({2, 6, 10}, {"axis1", "axis2", "axis3"});
  MeshAxesReplicaGroupList replica_group_across_axis2_subaxis(
      mesh_one_subaxis, {AxisRef(1, {2, 3})});
  MeshAxesReplicaGroupList replica_group_across_axis3_subaxis(
      mesh_one_subaxis, {AxisRef(2, {1, 2})});
  EXPECT_EQ(replica_group_across_axis2_subaxis.num_replica_groups(), 40);
  EXPECT_EQ(replica_group_across_axis2_subaxis.num_devices_per_group(), 3);
  EXPECT_EQ(replica_group_across_axis3_subaxis.num_replica_groups(), 60);
  EXPECT_EQ(replica_group_across_axis3_subaxis.num_devices_per_group(), 2);

  Mesh mesh_multiple_subaxis({2 * 3, 5 * 7, 11 * 13},
                             {"alpha", "beta", "gamma"});
  MeshAxesReplicaGroupList replica_group_across_multiple_subaxis1(
      mesh_multiple_subaxis,
      {AxisRef(0, {1, 2}), AxisRef(1, {1, 5}), AxisRef(2, {1, 11})});
  MeshAxesReplicaGroupList replica_group_across_multiple_subaxis2(
      mesh_multiple_subaxis,
      {AxisRef(0, {2, 3}), AxisRef(1, {5, 7}), AxisRef(2, {11, 13})});
  EXPECT_EQ(replica_group_across_multiple_subaxis1.num_replica_groups(),
            3 * 7 * 13);
  EXPECT_EQ(replica_group_across_multiple_subaxis1.num_devices_per_group(),
            2 * 5 * 11);
  EXPECT_EQ(replica_group_across_multiple_subaxis2.num_replica_groups(),
            2 * 5 * 11);
  EXPECT_EQ(replica_group_across_multiple_subaxis2.num_devices_per_group(),
            3 * 7 * 13);
}

TEST(MeshAxesReplicaGroupListTest, MeshAxesToString) {
  // No subaxes and iota device assignment.
  Mesh mesh_uvw({10, 12, 15}, {"u", "v", "w"});
  MeshAxesReplicaGroupList replica_group_across_uv(mesh_uvw,
                                                   {AxisRef(0), AxisRef(1)});
  EXPECT_EQ(replica_group_across_uv.ToString(), "@mesh<u=10,v=12,w=15> {u,v}");

  // Subaxes and replica group v2 iota style device assignment.
  Mesh mesh_abcd(
      TileAssignment(/*dims=*/{2, 4, 4, 2}, /*reshape_dims=*/{1, 4, 1, 16},
                     /*transpose_perm=*/{2, 3, 0, 1}),
      {"a", "b", "c", "d"});
  MeshAxesReplicaGroupList rg_abcd_across_multiple_axes_and_subaxes(
      mesh_abcd, {AxisRef(0), AxisRef(1, {1, 2}), AxisRef(3)});
  EXPECT_EQ(rg_abcd_across_multiple_axes_and_subaxes.ToString(),
            "@mesh<a=2,b=4,c=4,d=2>, device_ids=([4,16]T(1,0)) {a,b:(1)2,d}");

  // Subaxes and random device assignment.
  Array<int64_t> array({{8, 3, 7, 5, 4, 2, 6, 0, 1, 9}});
  array.Reshape({10});
  TileAssignment tile_assignment(std::make_shared<Array<int64_t>>(array));
  Mesh mesh_ooo(tile_assignment, {"ooo"});
  MeshAxesReplicaGroupList rg_ooo_across_ooo_5_2(mesh_ooo,
                                                 {AxisRef(0, {5, 2})});
  EXPECT_EQ(rg_ooo_across_ooo_5_2.ToString(),
            "@mesh<ooo=10>, device_ids=(8,3,7,5,4,2,6,0,1,9) {ooo:(5)2}");
}

TEST(MeshAxesReplicaGroupListTest, ValidatesIncompatibleAxes) {
  Mesh mesh({10}, {"u"});
  EXPECT_DEATH(
      {
        MeshAxesReplicaGroupList index_out_of_bounds(
            mesh, /*axes=*/{AxisRef(1, {1, 2})});
      },
      "Axis index must be less than number of axes");
  EXPECT_DEATH(
      {
        MeshAxesReplicaGroupList index_out_of_bounds(
            mesh, /*axes=*/{AxisRef(0, {8, 2})});
      },
      "Pre-size and size must divide the full axis size");
  EXPECT_DEATH(
      {
        MeshAxesReplicaGroupList index_out_of_bounds(
            mesh, /*axes=*/{AxisRef(0, {2, 8})});
      },
      "Pre-size and size must divide the full axis size");
  EXPECT_DEATH(
      {
        MeshAxesReplicaGroupList index_out_of_bounds(
            mesh, /*axes=*/{AxisRef(0, {1, 10})});
      },
      "Sub-axis size must be strictly less than the full axis size");
}

TEST(MeshAxesReplicaGroupListTest, ToReplicaGroupV2) {
  Mesh mesh_ab({6, 6}, {"a", "b"});

  // a:(1)3
  MeshAxesReplicaGroupList replica_group_a_1_3(mesh_ab, {AxisRef(0, {1, 3})});
  EXPECT_EQ(
      replica_group_a_1_3.flattened_replica_groups(),
      replica_group_a_1_3.ToIotaReplicaGroupList().flattened_replica_groups());

  // b:(3)2
  MeshAxesReplicaGroupList replica_group_b_3_2(mesh_ab, {AxisRef(1, {3, 2})});
  EXPECT_EQ(
      replica_group_b_3_2.flattened_replica_groups(),
      replica_group_b_3_2.ToIotaReplicaGroupList().flattened_replica_groups());

  // a:(1)2, b:(1)2
  MeshAxesReplicaGroupList replica_group_a_1_2_b_1_2(
      mesh_ab, {AxisRef(0, {1, 2}), AxisRef(1, {1, 2})});
  EXPECT_EQ(replica_group_a_1_2_b_1_2.flattened_replica_groups(),
            replica_group_a_1_2_b_1_2.ToIotaReplicaGroupList()
                .flattened_replica_groups());

  // a:(1)3, b:(1)3
  MeshAxesReplicaGroupList replica_group_a_1_3_b_1_3(
      mesh_ab, {AxisRef(0, {1, 3}), AxisRef(1, {1, 3})});
  EXPECT_EQ(replica_group_a_1_3_b_1_3.flattened_replica_groups(),
            replica_group_a_1_3_b_1_3.ToIotaReplicaGroupList()
                .flattened_replica_groups());

  // b:(1)3, a:(1)3 (Reverse order from above). This should produce the same
  // replica groups as the above but with ids in a different order.
  MeshAxesReplicaGroupList replica_group_b_1_3_a_1_3(
      mesh_ab, {AxisRef(1, {1, 3}), AxisRef(0, {1, 3})});
  EXPECT_EQ(replica_group_a_1_3_b_1_3.flattened_replica_groups(),
            replica_group_a_1_3_b_1_3.ToIotaReplicaGroupList()
                .flattened_replica_groups());

  Mesh mesh_cd({8, 6}, {"c", "d"});

  // c
  MeshAxesReplicaGroupList replica_group_c(mesh_cd, {AxisRef(0)});
  EXPECT_EQ(
      replica_group_c.flattened_replica_groups(),
      replica_group_c.ToIotaReplicaGroupList().flattened_replica_groups());

  // d
  MeshAxesReplicaGroupList replica_group_d(mesh_cd, {AxisRef(1)});
  EXPECT_EQ(
      replica_group_d.flattened_replica_groups(),
      replica_group_d.ToIotaReplicaGroupList().flattened_replica_groups());

  // c:(1)2, d:(4)2
  MeshAxesReplicaGroupList replica_group_c_1_2_c_4_2(
      mesh_cd, {AxisRef(0, {1, 2}), AxisRef(0, {4, 2})});
  EXPECT_EQ(replica_group_c_1_2_c_4_2.flattened_replica_groups(),
            replica_group_c_1_2_c_4_2.ToIotaReplicaGroupList()
                .flattened_replica_groups());

  // c:(2)3, d:(1)2
  MeshAxesReplicaGroupList replica_group_d_2_3_d_1_2(
      mesh_cd, {AxisRef(1, {2, 3}), AxisRef(1, {1, 2})});
  EXPECT_EQ(replica_group_d_2_3_d_1_2.flattened_replica_groups(),
            replica_group_d_2_3_d_1_2.ToIotaReplicaGroupList()
                .flattened_replica_groups());
}

TEST(MeshAxesReplicaGroupListTest, ToReplicaGroupV2WithComplexMesh) {
  Mesh mesh(TileAssignment(/*dims=*/{8, 2}, /*reshape_dims=*/{2, 4, 2},
                           /*transpose_perm=*/{2, 1, 0}),
            {"a", "b"});

  MeshAxesReplicaGroupList replica_group_a(mesh, {AxisRef(0)});
  EXPECT_EQ(replica_group_a.ToIotaReplicaGroupList().flattened_replica_groups(),
            replica_group_a.flattened_replica_groups());

  MeshAxesReplicaGroupList replica_group_b(mesh, {AxisRef(1)});
  EXPECT_EQ(replica_group_b.ToIotaReplicaGroupList().flattened_replica_groups(),
            replica_group_b.flattened_replica_groups());
}

TEST(MeshAxesReplicaGroupListTest, ToCollectiveDeviceList) {
  Mesh mesh({6, 6}, {"a", "b"});

  MeshAxesReplicaGroupList replica_group_b(mesh, {AxisRef(0)});
  EXPECT_EQ(
      replica_group_b.flattened_replica_groups(),
      replica_group_b.ToCollectiveDeviceList().flattened_replica_groups());

  MeshAxesReplicaGroupList replica_group_a_1_3(mesh, {AxisRef(0, {1, 3})});
  EXPECT_EQ(
      replica_group_a_1_3.flattened_replica_groups(),
      replica_group_a_1_3.ToCollectiveDeviceList().flattened_replica_groups());
}

TEST(CollectiveDeviceListTest, DefaultListToString) {
  EXPECT_EQ(CollectiveDeviceList().ToString(true), "{}");
  EXPECT_EQ(CollectiveDeviceList().ToString(false), "{}");

  ReplicaGroup empty_group;
  std::vector<ReplicaGroup> empty_groups;
  empty_groups.push_back(empty_group);
  empty_groups.push_back(empty_group);
  EXPECT_EQ(CollectiveDeviceList(empty_groups).ToString(), "{{},{}}");

  std::vector<std::vector<int64_t>> empty_groups2;
  EXPECT_EQ(CollectiveDeviceList(empty_groups2).ToString(), "{}");

  EXPECT_EQ(CollectiveDeviceList({{1}}).ToString(), "{{1}}");
  EXPECT_EQ(CollectiveDeviceList({{1, 2}, {3, 4}}).ToString(), "{{1,2},{3,4}}");
  EXPECT_EQ(CollectiveDeviceList({{1, 2, 3, 4, 5, 6, 7}}).ToString(),
            "{{1,2,3,4,5,6,7}}");
}

TEST(CollectiveDeviceListTest, DeepCopy) {
  CollectiveDeviceList orig({{1, 2, 3, 4}});
  CollectiveDeviceList copy = orig;
  EXPECT_EQ(&orig.replica_groups(), &copy.replica_groups());
  EXPECT_EQ(orig.ToString(), copy.ToString());
}

TEST(CollectiveDeviceListTest, DeepCopyIotaBeforeExpansion) {
  CollectiveDeviceList orig(IotaReplicaGroupList(2, 4));
  CollectiveDeviceList copy = orig;

  EXPECT_NE(&orig.iota_replica_group_list().value(),
            &copy.iota_replica_group_list().value());
  EXPECT_NE(&orig.replica_groups(), &copy.replica_groups());
  EXPECT_EQ(orig.ToString(), copy.ToString());
}

TEST(CollectiveDeviceListTest, DeepCopyIotaAfterExpansion) {
  CollectiveDeviceList orig(IotaReplicaGroupList(2, 4));
  const std::vector<ReplicaGroup>& local_ref = orig.replica_groups();
  CollectiveDeviceList copy = orig;

  EXPECT_NE(&orig.iota_replica_group_list().value(),
            &copy.iota_replica_group_list().value());
  EXPECT_EQ(&orig.replica_groups(), &copy.replica_groups());
  EXPECT_EQ(&local_ref, &copy.replica_groups());
  EXPECT_EQ(orig.ToString(), copy.ToString());
}

TEST(CollectiveDeviceListTest, DefaultListToProto) {
  EXPECT_THAT(CollectiveDeviceList().ToProto().replica_groups().size(), 0);
  CollectiveDeviceList list({{1, 2}, {3, 4}});
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_THAT(proto.replica_groups().size(), 2);
  EXPECT_THAT(proto.replica_groups(0).replica_ids(),
              testing::ElementsAre(1, 2));
  EXPECT_THAT(proto.replica_groups(1).replica_ids(),
              testing::ElementsAre(3, 4));
  EXPECT_FALSE(proto.has_iota_replica_group_list());
}

TEST(CollectiveDeviceListTest, DefaultListToProto2) {
  CollectiveDeviceList list({{1, 2, 3, 4, 5, 6, 7}});
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_THAT(proto.replica_groups().size(), 1);
  EXPECT_THAT(proto.replica_groups(0).replica_ids(),
              testing::ElementsAre(1, 2, 3, 4, 5, 6, 7));
  EXPECT_FALSE(proto.has_iota_replica_group_list());
}

TEST(CollectiveDeviceListTest, DefaultListFromProto) {
  HloInstructionProto initial_proto;
  *(initial_proto.mutable_collective_device_list()) =
      CreateDeviceListProto({{1, 2}, {3, 4}});
  CollectiveDeviceList list = CollectiveDeviceList::FromProto(initial_proto);
  EXPECT_EQ(list.replica_groups().size(), 2);
  EXPECT_THAT(list.replica_groups()[0].replica_ids(),
              testing::ElementsAre(1, 2));
  EXPECT_THAT(list.replica_groups()[1].replica_ids(),
              testing::ElementsAre(3, 4));
  EXPECT_FALSE(list.iota_replica_group_list().has_value());
}

TEST(CollectiveDeviceListTest, DefaultListFromProto2) {
  HloInstructionProto initial_proto;
  *(initial_proto.mutable_collective_device_list()) =
      CreateDeviceListProto({{1, 2, 3, 4, 5, 6, 7}});
  CollectiveDeviceList list = CollectiveDeviceList::FromProto(initial_proto);
  EXPECT_EQ(list.replica_groups().size(), 1);
  EXPECT_THAT(list.replica_groups()[0].replica_ids(),
              testing::ElementsAre(1, 2, 3, 4, 5, 6, 7));
  EXPECT_FALSE(list.iota_replica_group_list().has_value());
}

TEST(CollectiveDeviceListTest, IotaToString) {
  EXPECT_EQ(CollectiveDeviceList(IotaReplicaGroupList(0, 0)).ToString(),
            "[0,0]<=[0]");
  EXPECT_EQ(CollectiveDeviceList(IotaReplicaGroupList(2, 10)).ToString(),
            "[2,10]<=[20]");
}

TEST(CollectiveDeviceListTest, IotaToReplicaGroupString) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  EXPECT_EQ(list.ToString(false), "[2,10]<=[20]");
  EXPECT_EQ(list.ToString(true),
            "{{0,1,2,3,4,5,6,7,8,9},{10,11,12,13,14,15,16,17,18,19}}");
}

TEST(CollectiveDeviceListTest, IotaListToString2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  EXPECT_EQ(list.ToString(false), "[2,10]<=[4,5]T(1,0)");
  EXPECT_EQ(list.ToString(true),
            "{{0,5,10,15,1,6,11,16,2,7},{12,17,3,8,13,18,4,9,14,19}}");
}

TEST(CollectiveDeviceListTest, IotaListToProto) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_EQ(proto.iota_replica_group_list().num_replica_groups(), 2);
  EXPECT_EQ(proto.iota_replica_group_list().num_devices_per_group(), 10);
  EXPECT_THAT(proto.iota_replica_group_list().iota_reshape_dims(),
              testing::ElementsAre(20));
  EXPECT_THAT(proto.iota_replica_group_list().iota_transpose_perm(),
              testing::ElementsAre(0));
  EXPECT_THAT(proto.replica_groups_size(), 0);
}

TEST(CollectiveDeviceListTest, IotaListToProto2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  CollectiveDeviceListProto proto = list.ToProto();
  EXPECT_EQ(proto.iota_replica_group_list().num_replica_groups(), 2);
  EXPECT_EQ(proto.iota_replica_group_list().num_devices_per_group(), 10);
  EXPECT_THAT(proto.iota_replica_group_list().iota_reshape_dims(),
              testing::ElementsAre(4, 5));
  EXPECT_THAT(proto.iota_replica_group_list().iota_transpose_perm(),
              testing::ElementsAre(1, 0));
  EXPECT_THAT(proto.replica_groups_size(), 0);
}

TEST(CollectiveDeviceListTest, IotaListFromProto) {
  HloInstructionProto initial_proto;
  CollectiveDeviceListProto device_group;
  IotaReplicaGroupListProto* iota_replica_group_list =
      device_group.mutable_iota_replica_group_list();
  iota_replica_group_list->set_num_replica_groups(2);
  iota_replica_group_list->set_num_devices_per_group(10);
  iota_replica_group_list->add_iota_reshape_dims(20);
  iota_replica_group_list->add_iota_transpose_perm(0);
  *(initial_proto.mutable_collective_device_list()) = device_group;
  CollectiveDeviceList list = CollectiveDeviceList::FromProto(initial_proto);
  EXPECT_TRUE(list.iota_replica_group_list().has_value());
  EXPECT_EQ(list.iota_replica_group_list()->num_replica_groups(), 2);
  EXPECT_EQ(list.iota_replica_group_list()->num_devices_per_group(), 10);
  EXPECT_THAT(list.iota_replica_group_list()->reshape_dims(),
              testing::ElementsAre(20));
  EXPECT_THAT(list.iota_replica_group_list()->transpose_perm(),
              testing::ElementsAre(0));
}

TEST(CollectiveDeviceListTest, IotaListFromProto2) {
  HloInstructionProto initial_proto;
  CollectiveDeviceListProto device_group;
  IotaReplicaGroupListProto* iota_replica_group_list =
      device_group.mutable_iota_replica_group_list();
  iota_replica_group_list->set_num_replica_groups(2);
  iota_replica_group_list->set_num_devices_per_group(10);
  iota_replica_group_list->add_iota_reshape_dims(4);
  iota_replica_group_list->add_iota_reshape_dims(5);
  iota_replica_group_list->add_iota_transpose_perm(1);
  iota_replica_group_list->add_iota_transpose_perm(0);
  *(initial_proto.mutable_collective_device_list()) = device_group;
  CollectiveDeviceList list = CollectiveDeviceList::FromProto(initial_proto);
  EXPECT_TRUE(list.iota_replica_group_list().has_value());
  EXPECT_EQ(list.iota_replica_group_list()->num_replica_groups(), 2);
  EXPECT_EQ(list.iota_replica_group_list()->num_devices_per_group(), 10);
  EXPECT_THAT(list.iota_replica_group_list()->reshape_dims(),
              testing::ElementsAre(4, 5));
  EXPECT_THAT(list.iota_replica_group_list()->transpose_perm(),
              testing::ElementsAre(1, 0));
}

}  // namespace xla
