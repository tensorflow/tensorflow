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

TEST(CartesianProductTest, NonEmptyCartesianProduct) {
  CartesianProduct product_ascending({1, 2, 3});
  std::vector<std::vector<int64_t>> expected_ascending = {
      {0, 0, 0}, {0, 0, 1}, {0, 0, 2}, {0, 1, 0}, {0, 1, 1}, {0, 1, 2}};
  int cp_count = 0;
  for (std::vector<int64_t> current : product_ascending) {
    EXPECT_THAT(current,
                testing::ElementsAreArray(expected_ascending[cp_count]));
    cp_count++;
  }
  EXPECT_EQ(cp_count, expected_ascending.size());

  CartesianProduct product_descending({4, 2});
  std::vector<std::vector<int>> expected_descending = {
      {0, 0}, {0, 1}, {1, 0}, {1, 1}, {2, 0}, {2, 1}, {3, 0}, {3, 1}};
  cp_count = 0;
  for (std::vector<int64_t> current : product_descending) {
    EXPECT_THAT(current,
                testing::ElementsAreArray(expected_descending[cp_count]));
    cp_count++;
  }
  EXPECT_EQ(cp_count, expected_descending.size());

  CartesianProduct product_ones({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  std::vector<std::vector<int64_t>> expected_ones = {
      {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  cp_count = 0;
  for (std::vector<int64_t> current : product_ones) {
    EXPECT_THAT(current, testing::ElementsAreArray(expected_ones[cp_count]));
    cp_count++;
  }
  EXPECT_EQ(cp_count, expected_ones.size());
}

TEST(CartesianProductTest, EmptyCartesianProduct) {
  CartesianProduct product_empty({});
  EXPECT_TRUE(product_empty.begin().at_end);
  for (  // NOLINT(clang-diagnostic-unreachable-code-loop-increment
      std::vector<int64_t> current : product_empty) {
    FAIL() << "Expected empty product to be empty";
  }

  CartesianProduct product_empty_axis({1, 2, 3, 4, 5, 0, 6, 7});
  EXPECT_TRUE(product_empty_axis.begin().at_end);
  for (  // NOLINT(clang-diagnostic-unreachable-code-loop-increment
      std::vector<int64_t> current : product_empty_axis) {
    FAIL() << "Expected product of empty axis to be empty";
  }
}

TEST(MeshAxesReplicaGroupListTest, MaterializedReplicaGroups) {
  Mesh mesh_xy(TileAssignment(IotaTileAssignment::Create(
                   /*dims=*/{2, 2})),
               /*axes_names=*/{"x", "y"});

  MeshAxesReplicaGroupList replica_group_reduce_none(mesh_xy, /*axes=*/{});
  std::vector<std::vector<int64_t>> expected_replica_groups_none = {
      {0}, {1}, {2}, {3}};
  EXPECT_THAT(replica_group_reduce_none.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_none));

  MeshAxesReplicaGroupList replica_group_reduce_x(mesh_xy,
                                                  /*axes=*/{AxisRef(0)});
  std::vector<std::vector<int64_t>> expected_replica_groups_x = {{0, 2},
                                                                 {1, 3}};
  EXPECT_THAT(replica_group_reduce_x.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_x));

  MeshAxesReplicaGroupList replica_group_reduce_y(mesh_xy,
                                                  /*axes=*/{AxisRef(1)});
  std::vector<std::vector<int64_t>> expected_replica_groups_y = {{0, 1},
                                                                 {2, 3}};
  EXPECT_THAT(replica_group_reduce_y.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_y));

  MeshAxesReplicaGroupList replica_group_reduce_xy(
      mesh_xy, /*axes=*/{AxisRef(0), AxisRef(1)});
  std::vector<std::vector<int64_t>> expected_replica_groups_xy = {{0, 1, 2, 3}};
  EXPECT_THAT(replica_group_reduce_xy.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_xy));
}

TEST(MeshAxesReplicaGroupListTest, MaterializedReplicaGroupsWithSubaxes) {
  Mesh mesh(TileAssignment(IotaTileAssignment::Create(
                /*dims=*/{6, 6})),
            /*axes_names=*/{"a", "b"});

  // a:(1)2
  MeshAxesReplicaGroupList replica_group_a_1_2(mesh,
                                               /*axes=*/{AxisRef(0, {1, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_2 = {
      {0, 18},  {1, 19},  {2, 20},  {3, 21},  {4, 22},  {5, 23},
      {6, 24},  {7, 25},  {8, 26},  {9, 27},  {10, 28}, {11, 29},
      {12, 30}, {13, 31}, {14, 32}, {15, 33}, {16, 34}, {17, 35}};
  EXPECT_THAT(
      replica_group_a_1_2.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_a_1_2));

  // a:(1)3
  MeshAxesReplicaGroupList replica_group_a_1_3(mesh,
                                               /*axes=*/{AxisRef(0, {1, 3})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_3 = {
      {0, 12, 24}, {1, 13, 25}, {2, 14, 26},  {3, 15, 27},
      {4, 16, 28}, {5, 17, 29}, {6, 18, 30},  {7, 19, 31},
      {8, 20, 32}, {9, 21, 33}, {10, 22, 34}, {11, 23, 35}};
  EXPECT_THAT(
      replica_group_a_1_3.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_a_1_3));

  // a:(3)2
  MeshAxesReplicaGroupList replica_group_a_3_2(mesh,
                                               /*axes=*/{AxisRef(0, {3, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_3_2 = {
      {0, 6},   {1, 7},   {2, 8},   {3, 9},   {4, 10},  {5, 11},
      {12, 18}, {13, 19}, {14, 20}, {15, 21}, {16, 22}, {17, 23},
      {24, 30}, {25, 31}, {26, 32}, {27, 33}, {28, 34}, {29, 35}};
  EXPECT_THAT(
      replica_group_a_3_2.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_a_3_2));

  // b:(1)2
  MeshAxesReplicaGroupList replica_group_b_1_2(mesh,
                                               /*axes=*/{AxisRef(1, {1, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_b_1_2 = {
      {0, 3},   {1, 4},   {2, 5},   {6, 9},   {7, 10},  {8, 11},
      {12, 15}, {13, 16}, {14, 17}, {18, 21}, {19, 22}, {20, 23},
      {24, 27}, {25, 28}, {26, 29}, {30, 33}, {31, 34}, {32, 35}};
  EXPECT_THAT(
      replica_group_b_1_2.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_b_1_2));

  // b:(1)3
  MeshAxesReplicaGroupList replica_group_b_1_3(mesh,
                                               /*axes=*/{AxisRef(1, {1, 3})});
  std::vector<std::vector<int64_t>> expected_replica_groups_b_1_3 = {
      {0, 2, 4},    {1, 3, 5},    {6, 8, 10},   {7, 9, 11},
      {12, 14, 16}, {13, 15, 17}, {18, 20, 22}, {19, 21, 23},
      {24, 26, 28}, {25, 27, 29}, {30, 32, 34}, {31, 33, 35}};
  EXPECT_THAT(
      replica_group_b_1_3.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_b_1_3));

  // b:(3)2
  MeshAxesReplicaGroupList replica_group_b_3_2(mesh,
                                               /*axes=*/{AxisRef(1, {3, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_b_3_2 = {
      {0, 1},   {2, 3},   {4, 5},   {6, 7},   {8, 9},   {10, 11},
      {12, 13}, {14, 15}, {16, 17}, {18, 19}, {20, 21}, {22, 23},
      {24, 25}, {26, 27}, {28, 29}, {30, 31}, {32, 33}, {34, 35}};
  EXPECT_THAT(
      replica_group_b_3_2.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_b_3_2));

  // a:(1)2, b:(1)2
  MeshAxesReplicaGroupList replica_group_a_1_2_b_1_2(
      mesh,
      /*axes=*/{AxisRef(0, {1, 2}), AxisRef(1, {1, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_2_b_1_2 = {
      {0, 3, 18, 21},   {1, 4, 19, 22},   {2, 5, 20, 23},
      {6, 9, 24, 27},   {7, 10, 25, 28},  {8, 11, 26, 29},
      {12, 15, 30, 33}, {13, 16, 31, 34}, {14, 17, 32, 35}};
  EXPECT_THAT(
      replica_group_a_1_2_b_1_2.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_a_1_2_b_1_2));

  // a:(1)3, b:(1)3
  MeshAxesReplicaGroupList replica_group_a_1_3_b_1_3(
      mesh,
      /*axes=*/{AxisRef(0, {1, 3}), AxisRef(1, {1, 3})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_1_3_b_1_3 = {
      {0, 2, 4, 12, 14, 16, 24, 26, 28},
      {1, 3, 5, 13, 15, 17, 25, 27, 29},
      {6, 8, 10, 18, 20, 22, 30, 32, 34},
      {7, 9, 11, 19, 21, 23, 31, 33, 35}};
  EXPECT_THAT(
      replica_group_a_1_3_b_1_3.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_a_1_3_b_1_3));

  // a:(3)2, b:(3)2
  MeshAxesReplicaGroupList replica_group_a_3_2_b_3_2(
      mesh,
      /*axes=*/{AxisRef(0, {3, 2}), AxisRef(1, {3, 2})});
  std::vector<std::vector<int64_t>> expected_replica_groups_a_3_2_b_3_2 = {
      {0, 1, 6, 7},     {2, 3, 8, 9},     {4, 5, 10, 11},
      {12, 13, 18, 19}, {14, 15, 20, 21}, {16, 17, 22, 23},
      {24, 25, 30, 31}, {26, 27, 32, 33}, {28, 29, 34, 35}};
  EXPECT_THAT(
      replica_group_a_3_2_b_3_2.flattened_replica_groups(),
      testing::UnorderedElementsAreArray(expected_replica_groups_a_3_2_b_3_2));
}

TEST(MeshAxesReplicaGroupListTest, MaterializedReplicaGroupsMatchExpectedV2) {
  Mesh mesh(TileAssignment(IotaTileAssignment::Create(/*dims=*/{8})),
            /*axes_names=*/{"a"});

  // a:(1)2 -> replica_groups=[4,2]<=[2,4]T(1,0)
  MeshAxesReplicaGroupList v3_subaxis_1_2(mesh, /*axes=*/{AxisRef(0, {1, 2})});
  IotaReplicaGroupList v2_subaxis_1_2(4, 2, {2, 4}, {1, 0});
  EXPECT_THAT(v3_subaxis_1_2.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(
                  v2_subaxis_1_2.flattened_replica_groups()));

  // a:(1)4 -> replica_groups=[2,4]<=[4,2]T(1,0)
  MeshAxesReplicaGroupList v3_subaxis_1_4(mesh, /*axes=*/{AxisRef(0, {1, 4})});
  IotaReplicaGroupList v2_subaxis_1_4(2, 4, {4, 2}, {1, 0});
  EXPECT_THAT(v3_subaxis_1_4.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(
                  v2_subaxis_1_4.flattened_replica_groups()));

  // a:(2)2 -> replica_groups=[4,2]<=[2,2,2]T(0,2,1)
  MeshAxesReplicaGroupList v3_subaxis_2_2(mesh, /*axes=*/{AxisRef(0, {2, 2})});
  IotaReplicaGroupList v2_subaxis_2_2(4, 2, {2, 2, 2}, {0, 2, 1});
  EXPECT_THAT(v3_subaxis_2_2.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(
                  v2_subaxis_2_2.flattened_replica_groups()));

  // a:(2)4 -> replica_groups=[2,4]<=[8]
  MeshAxesReplicaGroupList v3_subaxis_2_4(mesh, /*axes=*/{AxisRef(0, {2, 4})});
  IotaReplicaGroupList v2_subaxis_2_4(2, 4, {8}, {0});
  EXPECT_THAT(v3_subaxis_2_4.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(
                  v2_subaxis_2_4.flattened_replica_groups()));

  // a:(4)2 -> replica_groups=[4,2]<=[8]
  MeshAxesReplicaGroupList v3_subaxis_4_2(mesh, /*axes=*/{AxisRef(0, {4, 2})});
  IotaReplicaGroupList v2_subaxis_4_2(4, 2, {8}, {0});
  EXPECT_THAT(v3_subaxis_4_2.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(
                  v2_subaxis_4_2.flattened_replica_groups()));

  // a      -> replica_groups=[1,8]<=[8]
  MeshAxesReplicaGroupList v3_no_subaxis(mesh, /*axes=*/{AxisRef(0)});
  IotaReplicaGroupList v2_no_subaxis(1, 8, {8}, {0});
  EXPECT_THAT(v3_no_subaxis.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(
                  v2_no_subaxis.flattened_replica_groups()));
}

TEST(MeshAxesReplicaGroupListTest,
     MaterializedReplicaGroupsRespectNonIotaDeviceOrdering) {
  // Create a mesh with non-iota device ordering.
  Array2D<int64_t> array({{3, 1}, {0, 2}});
  TileAssignment tile_assignment(std::make_shared<Array<int64_t>>(array));
  Mesh mesh_xy(tile_assignment, /*axes_names=*/{"x", "y"});

  // Reduce along x axis.
  MeshAxesReplicaGroupList replica_group_reduce_x(mesh_xy,
                                                  /*axes=*/{AxisRef(0)});
  // With iota device ordering, the expected replica groups would be
  // {{0, 2}, {1, 3}}.
  std::vector<std::vector<int64_t>> expected_replica_groups_x = {{3, 0},
                                                                 {1, 2}};
  EXPECT_THAT(replica_group_reduce_x.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_x));

  // Reduce along y axis.
  MeshAxesReplicaGroupList replica_group_reduce_y(mesh_xy,
                                                  /*axes=*/{AxisRef(1)});
  // With iota device ordering, the expected replica groups would be
  // {{0, 1}, {2, 3}}.
  std::vector<std::vector<int64_t>> expected_replica_groups_y = {{3, 1},
                                                                 {0, 2}};
  EXPECT_THAT(replica_group_reduce_y.flattened_replica_groups(),
              testing::UnorderedElementsAreArray(expected_replica_groups_y));
}

TEST(MeshAxesReplicaGroupListTest, NumReplicaGroups) {
  Mesh mesh_reduce_all_axes(TileAssignment(IotaTileAssignment::Create(
                                /*dims=*/{4, 4})),
                            /*axes_names=*/{"x", "y"});
  MeshAxesReplicaGroupList replica_group_across_all_axes(
      mesh_reduce_all_axes,
      /*axes=*/{AxisRef(0), AxisRef(1)});
  EXPECT_EQ(replica_group_across_all_axes.num_replica_groups(), 1);
  EXPECT_EQ(replica_group_across_all_axes.num_devices_per_group(), 16);

  Mesh mesh_reduce_one_axes(TileAssignment(IotaTileAssignment::Create(
                                /*dims=*/{3, 5})),
                            /*axes_names=*/{"a", "b"});
  MeshAxesReplicaGroupList replica_group_across_a(mesh_reduce_one_axes,
                                                  /*axes=*/{AxisRef(0)});
  MeshAxesReplicaGroupList replica_group_across_b(mesh_reduce_one_axes,
                                                  /*axes=*/{AxisRef(1)});
  EXPECT_EQ(replica_group_across_a.num_replica_groups(), 5);
  EXPECT_EQ(replica_group_across_a.num_devices_per_group(), 3);
  EXPECT_EQ(replica_group_across_b.num_replica_groups(), 3);
  EXPECT_EQ(replica_group_across_b.num_devices_per_group(), 5);

  Mesh mesh_reduce_no_axes(TileAssignment(IotaTileAssignment::Create(
                               /*dims=*/{2, 3, 5})),
                           /*axes_names=*/{"p1", "p2", "p3"});
  MeshAxesReplicaGroupList replica_group_across_no_axes(mesh_reduce_no_axes,
                                                        /*axes=*/{});
  EXPECT_EQ(replica_group_across_no_axes.num_replica_groups(), 2 * 3 * 5);
  EXPECT_EQ(replica_group_across_no_axes.num_devices_per_group(), 1);
}

TEST(MeshAxesReplicaGroupListTest, ReplicaGroupsCountAndSizeForSubaxes) {
  Mesh mesh_reduce_one_subaxis(TileAssignment(IotaTileAssignment::Create(
                                   /*dims=*/{2, 6, 10})),
                               /*axes_names=*/{"axis1", "axis2", "axis3"});
  MeshAxesReplicaGroupList replica_group_across_axis1_subaxis(
      mesh_reduce_one_subaxis,
      /*axes=*/{AxisRef(0, {1, 2})});
  MeshAxesReplicaGroupList replica_group_across_axis2_subaxis(
      mesh_reduce_one_subaxis,
      /*axes=*/{AxisRef(1, {2, 3})});
  EXPECT_EQ(replica_group_across_axis1_subaxis.num_replica_groups(), 60);
  EXPECT_EQ(replica_group_across_axis1_subaxis.num_devices_per_group(), 2);
  EXPECT_EQ(replica_group_across_axis2_subaxis.num_replica_groups(), 40);
  EXPECT_EQ(replica_group_across_axis2_subaxis.num_devices_per_group(), 3);

  Mesh mesh_reduce_multiple_subaxis(TileAssignment(IotaTileAssignment::Create(
                                        /*dims=*/{2 * 3, 5 * 7, 11 * 13})),
                                    /*axes_names=*/{"alpha", "beta", "gamma"});
  MeshAxesReplicaGroupList replica_group_across_multiple_subaxis1(
      mesh_reduce_multiple_subaxis,
      /*axes=*/{AxisRef(0, {1, 2}), AxisRef(1, {1, 5}), AxisRef(2, {1, 11})});
  MeshAxesReplicaGroupList replica_group_across_multiple_subaxis2(
      mesh_reduce_multiple_subaxis,
      /*axes=*/{AxisRef(0, {2, 3}), AxisRef(1, {5, 7}), AxisRef(2, {11, 13})});
  EXPECT_EQ(replica_group_across_multiple_subaxis1.num_replica_groups(), 273);
  EXPECT_EQ(replica_group_across_multiple_subaxis1.num_devices_per_group(),
            2 * 5 * 11);
  EXPECT_EQ(replica_group_across_multiple_subaxis2.num_replica_groups(), 110);
  EXPECT_EQ(replica_group_across_multiple_subaxis2.num_devices_per_group(),
            3 * 7 * 13);
}

TEST(MeshAxesReplicaGroupListTest, MeshAxesToString) {
  // No subaxes and iota device assignment.
  Mesh mesh_uvw(TileAssignment(IotaTileAssignment::Create(
                    /*dims=*/{10, 12, 15})),
                /*axes_names=*/{"u", "v", "w"});
  MeshAxesReplicaGroupList replica_group_across_none(mesh_uvw, /*axes=*/{});
  EXPECT_EQ(replica_group_across_none.ToString(), "@mesh<u=10,v=12,w=15> {}");
  MeshAxesReplicaGroupList replica_group_across_uv(
      mesh_uvw,
      /*axes=*/{AxisRef(0), AxisRef(1)});
  EXPECT_EQ(replica_group_across_uv.ToString(), "@mesh<u=10,v=12,w=15> {u,v}");

  // Subaxes and replica group v2 iota style device assignment.
  Mesh mesh_abcd(TileAssignment(IotaTileAssignment::Create(
                     /*dims=*/{2, 4, 4, 2}, /*reshape_dims=*/{1, 4, 1, 16},
                     /*transpose_perm=*/{2, 3, 0, 1})),
                 /*axes_names=*/{"a", "b", "c", "d"});
  MeshAxesReplicaGroupList rg_abcd_across_none(mesh_abcd, /*axes=*/{});
  EXPECT_EQ(rg_abcd_across_none.ToString(),
            "@mesh<a=2,b=4,c=4,d=2> {}, devices=[2,4,4,2]<=[4,16]T(1,0)");
  MeshAxesReplicaGroupList rg_abcd_across_multiple_axes_and_subaxes(
      mesh_abcd, /*axes=*/{AxisRef(0), AxisRef(1, {1, 2}), AxisRef(3)});
  EXPECT_EQ(
      rg_abcd_across_multiple_axes_and_subaxes.ToString(),
      "@mesh<a=2,b=4,c=4,d=2> {a,b:(1)2,d}, devices=[2,4,4,2]<=[4,16]T(1,0)");

  // Subaxes and random device assignment.
  Array<int64_t> array({{8, 3, 7, 5, 4, 2, 6, 0, 1, 9}});
  array.Reshape({10});
  TileAssignment tile_assignment(std::make_shared<Array<int64_t>>(array));
  Mesh mesh_ooo(tile_assignment, /*axes_names=*/{"ooo"});
  MeshAxesReplicaGroupList rg_ooo_across_none(mesh_ooo, /*axes=*/{});
  EXPECT_EQ(rg_ooo_across_none.ToString(),
            "@mesh<ooo=10> {}, devices=[10]8,3,7,5,4,2,6,0,1,9");
  MeshAxesReplicaGroupList rg_ooo_across_ooo_5_2(mesh_ooo,
                                                 /*axes=*/{AxisRef(0, {5, 2})});
  EXPECT_EQ(rg_ooo_across_ooo_5_2.ToString(),
            "@mesh<ooo=10> {ooo:(5)2}, devices=[10]8,3,7,5,4,2,6,0,1,9");
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
