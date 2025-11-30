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

#include "xla/service/spmd/spmd_partitioner_util.h"

#include <cstdint>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/ir/tile_assignment.h"

namespace xla {
namespace spmd {
namespace {

TEST(SPMDPartitionerUtilTest, PartialReplicateReshardCompatibleSharding1) {
  HloSharding partial_sharding =
      HloSharding::PartialTile(TileAssignment({1, 2, 2}));
  const std::vector<HloSharding> target_shardings = {
      HloSharding::IotaTile({2, 2}),
      HloSharding::IotaTile({2, 2}, {2, 2}, {1, 0})};
  for (const auto& target_sharding : target_shardings) {
    auto result = PartialReplicateReshardCompatibleSharding(partial_sharding,
                                                            target_sharding);
    EXPECT_EQ(result, target_shardings[1]);
  }

  partial_sharding =
      HloSharding::PartialTile(TileAssignment({1, 2, 2}, {2, 2}, {1, 0}));
  for (const auto& target_sharding : target_shardings) {
    auto result = PartialReplicateReshardCompatibleSharding(partial_sharding,
                                                            target_sharding);
    EXPECT_EQ(result, target_shardings[0]);
  }
}

TEST(SPMDPartitionerUtilTest, PartialReplicateReshardCompatibleSharding2) {
  HloSharding partial_sharding =
      HloSharding::PartialTile(TileAssignment({2, 2, 8}));
  const std::vector<HloSharding> target_shardings = {
      HloSharding::PartialTile(
          TileAssignment({4, 4, 2}, {2, 2, 2, 2, 2}, {0, 2, 1, 3, 4})),
      HloSharding::PartialTile(
          TileAssignment({4, 4, 2}, {2, 2, 2, 2, 2}, {0, 2, 1, 4, 3})),
      HloSharding::PartialTile(
          TileAssignment({4, 4, 2}, {2, 2, 2, 2, 2}, {0, 3, 1, 2, 4})),
      HloSharding::PartialTile(
          TileAssignment({4, 4, 2}, {2, 2, 2, 2, 2}, {0, 3, 1, 4, 2})),
      HloSharding::PartialTile(
          TileAssignment({4, 4, 2}, {2, 2, 2, 2, 2}, {0, 4, 1, 2, 3})),
      HloSharding::PartialTile(
          TileAssignment({4, 4, 2}, {2, 2, 2, 2, 2}, {0, 4, 1, 3, 2}))};
  for (const auto& target_sharding : target_shardings) {
    auto result = PartialReplicateReshardCompatibleSharding(partial_sharding,
                                                            target_sharding);
    EXPECT_EQ(result, target_sharding);
  }
}

TEST(SPMDPartitionerUtilTest, GetPartitionGroupsForReplication) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2});
  std::vector<std::vector<int64_t>> actual_partition_groups =
      GetPartitionGroupsForReplication(sharding, {1});
  std::vector<std::vector<int64_t>> expected_partition_groups = {
      {0, 2}, {1, 3}, {4, 6}, {5, 7}};
  EXPECT_THAT(actual_partition_groups,
              testing::ContainerEq(expected_partition_groups));
}

TEST(SPMDPartitionerUtilTest, GetPartitionGroupsForReplication2) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2}, {2, 2, 2}, {0, 2, 1});
  std::vector<std::vector<int64_t>> actual_partition_groups =
      GetPartitionGroupsForReplication(sharding, {0, 2});
  std::vector<std::vector<int64_t>> expected_partition_groups = {{0, 2, 4, 6},
                                                                 {1, 3, 5, 7}};
  EXPECT_THAT(actual_partition_groups,
              testing::ContainerEq(expected_partition_groups));
}

TEST(SPMDPartitionerUtilTest, GetIotaPartitionGroupsAcrossTargetDims) {
  HloSharding sharding = HloSharding::IotaTile({8, 8, 16});
  std::optional<IotaReplicaGroupList> actual_partition_group_list =
      GetIotaPartitionGroupsAcrossTargetDims(sharding, {0, 1}, {4, 4});
  EXPECT_TRUE(actual_partition_group_list.has_value());
  EXPECT_EQ(actual_partition_group_list->num_replica_groups(), 64);
  EXPECT_EQ(actual_partition_group_list->num_devices_per_group(), 16);
  EXPECT_THAT(actual_partition_group_list->reshape_dims(),
              testing::ElementsAre(2, 4, 2, 4, 16));
  EXPECT_THAT(actual_partition_group_list->transpose_perm(),
              testing::ElementsAre(0, 2, 4, 1, 3));
}

TEST(SPMDPartitionerUtilTest, GetIotaPartitionGroupsForReplication) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2});
  std::optional<IotaReplicaGroupList> actual_partition_group_list =
      GetIotaPartitionGroupsForReplication(sharding, {1});
  EXPECT_TRUE(actual_partition_group_list.has_value());
  EXPECT_EQ(actual_partition_group_list->num_replica_groups(), 4);
  EXPECT_EQ(actual_partition_group_list->num_devices_per_group(), 2);
  EXPECT_THAT(actual_partition_group_list->reshape_dims(),
              testing::ElementsAre(2, 2, 2));
  EXPECT_THAT(actual_partition_group_list->transpose_perm(),
              testing::ElementsAre(0, 2, 1));
}

TEST(SPMDPartitionerUtilTest, GetIotaPartitionGroupsForReplication2) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2}, {2, 2, 2}, {0, 2, 1});
  std::optional<IotaReplicaGroupList> actual_partition_group_list =
      GetIotaPartitionGroupsForReplication(sharding, {0, 2});
  EXPECT_TRUE(actual_partition_group_list.has_value());
  EXPECT_EQ(actual_partition_group_list->num_replica_groups(), 2);
  EXPECT_EQ(actual_partition_group_list->num_devices_per_group(), 4);
  EXPECT_THAT(actual_partition_group_list->reshape_dims(),
              testing::ElementsAre(4, 2));
  EXPECT_THAT(actual_partition_group_list->transpose_perm(),
              testing::ElementsAre(1, 0));
}

TEST(SPMDPartitionerUtilTest, ExpandPartitionGroupListAcrossReplicas) {
  IotaReplicaGroupList partition_group_list =
      IotaReplicaGroupList(10, 5, {2, 5, 5}, {0, 2, 1});
  IotaReplicaGroupList expanded_partition_group_list =
      ExpandPartitionGroupListAcrossReplicas(partition_group_list, 2, 50)
          .iota_replica_group_list()
          .value();
  EXPECT_EQ(expanded_partition_group_list.num_replica_groups(), 20);
  EXPECT_EQ(expanded_partition_group_list.num_devices_per_group(), 5);
  EXPECT_THAT(expanded_partition_group_list.reshape_dims(),
              testing::ElementsAre(4, 5, 5));
  EXPECT_THAT(expanded_partition_group_list.transpose_perm(),
              testing::ElementsAre(0, 2, 1));
}

TEST(SPMDPartitionerUtilDeathTest, ExpandPartitionGroupListAcrossReplicas) {
  IotaReplicaGroupList partition_group_list =
      IotaReplicaGroupList(10, 5, {2, 5, 5}, {0, 2, 1});
  // If we try to expand partition group list across replicas for a partition
  // group list that does not cover all available partitions, we should exit
  // with a failure.
  ASSERT_DEATH(
      {
        auto expanded_partition_group_list =
            ExpandPartitionGroupListAcrossReplicas(partition_group_list, 2, 60);
      },
      "Check failed: \\(partition_group_count \\* partition_group_size\\) == "
      "num_partitions \\(50 vs\\. 60\\)");
}

TEST(SPMDPartitionerUtilTest, GetMeshAxesPartitionGroupsAcrossTargetDims) {
  // V2 Sharding
  HloSharding sharding_v2 = HloSharding::IotaTile({8, 8, 16});
  std::optional<MeshAxesReplicaGroupList> v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding_v2, {0, 1}, {4, 4});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 64);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 16);
  EXPECT_EQ(v3_group_list->ToString(),
            "@mesh<axis_0=8,axis_1=8,axis_2=16> {axis_0:(2)4,axis_1:(2)4}");

  // V3 Sharding (Will correctly reflect the real mesh axis names)
  NamedSharding named_sharding(Mesh({8, 8, 16}, {"good", "bad", "ugly"}));
  HloSharding sharding_v3 = HloSharding::FromNamedSharding(named_sharding);
  v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding_v3, {0, 1}, {4, 4});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 64);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 16);
  EXPECT_EQ(v3_group_list->ToString(),
            "@mesh<good=8,bad=8,ugly=16> {good:(2)4,bad:(2)4}");
}

TEST(SPMDPartitionerUtilTest, GetMeshAxesPartitionGroupsForReplication) {
  // V2 Sharding
  HloSharding sharding_v2 = HloSharding::IotaTile({2, 2, 2});
  std::optional<MeshAxesReplicaGroupList> v3_group_list =
      GetMeshAxesPartitionGroupsForReplication(sharding_v2, {1});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 4);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 2);
  EXPECT_EQ(v3_group_list->ToString(),
            "@mesh<axis_0=2,axis_1=2,axis_2=2> {axis_1}");

  // V3 Sharding (Will correctly reflect the real mesh axis names)
  NamedSharding named_sharding(Mesh({2, 2, 2}, {"Euler", "Gauss", "Riemann"}));
  HloSharding sharding_v3 = HloSharding::FromNamedSharding(named_sharding);
  v3_group_list = GetMeshAxesPartitionGroupsForReplication(sharding_v3, {1});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 4);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 2);
  EXPECT_EQ(v3_group_list->ToString(),
            "@mesh<Euler=2,Gauss=2,Riemann=2> {Gauss}");
}

TEST(SPMDPartitionerUtilTest, ReturnNulloptForEmptyReplicationDims) {
  HloSharding sharding_v2 = HloSharding::IotaTile({3, 5, 7});
  {
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsAcrossTargetDims(sharding_v2, {}, {});
    EXPECT_FALSE(v3_group_list.has_value());
  }
  {
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsForReplication(sharding_v2, {});
    EXPECT_FALSE(v3_group_list.has_value());
  }
}

TEST(SPMDPartitionerUtilDeathTest, V3PartitionGroupsFailForInvalidInputs) {
  HloSharding sharding_v2 =
      HloSharding::IotaTile({2, 3, 6, 11, 23, 47, 106, 235});
  ASSERT_DEATH(
      {
        std::optional<MeshAxesReplicaGroupList> v3_group_list =
            GetMeshAxesPartitionGroupsAcrossTargetDims(sharding_v2, {0, 3, 6},
                                                       {2, 2});
      },
      "target_dims and group_sizes must");
}

TEST(SPMDPartitionerUtilTest,
     ValidateEqualityOfV2AndV3ReplicaGroupsAcrossTargetDims) {
  auto v2AndV3ReplicaGroupsMatch =
      [](std::optional<IotaReplicaGroupList> v2_group_list,
         std::optional<MeshAxesReplicaGroupList> v3_group_list,
         int64_t expected_num_replica_groups,
         int64_t expected_num_devices_per_group) {
        EXPECT_TRUE(v2_group_list.has_value());
        EXPECT_TRUE(v3_group_list.has_value());
        EXPECT_EQ(v2_group_list->num_replica_groups(),
                  expected_num_replica_groups);
        EXPECT_EQ(v3_group_list->num_replica_groups(),
                  expected_num_replica_groups);
        EXPECT_EQ(v2_group_list->num_devices_per_group(),
                  expected_num_devices_per_group);
        EXPECT_EQ(v3_group_list->num_devices_per_group(),
                  expected_num_devices_per_group);
        EXPECT_EQ(v2_group_list->flattened_replica_groups(),
                  v3_group_list->flattened_replica_groups());
      };
  {
    HloSharding sharding = HloSharding::IotaTile({32, 32});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsAcrossTargetDims(sharding, {0}, {16});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0}, {16});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/(32 * 32) / (16),
                              /*expected_num_devices_per_group=*/16);
  }
  {
    HloSharding sharding = HloSharding::IotaTile({16, 64});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsAcrossTargetDims(sharding, {1}, {32});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {1}, {32});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/(16 * 64) / (32),
                              /*expected_num_devices_per_group=*/32);
  }
  {
    HloSharding sharding = HloSharding::IotaTile({8, 8, 16});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsAcrossTargetDims(sharding, {0, 1}, {4, 4});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0, 1}, {4, 4});
    v2AndV3ReplicaGroupsMatch(
        v2_group_list, v3_group_list,
        /*expected_num_replica_groups=*/(8 * 8 * 16) / (4 * 4),
        /*expected_num_devices_per_group=*/4 * 4);
  }
  {
    HloSharding sharding = HloSharding::IotaTile({16, 8, 16});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsAcrossTargetDims(sharding, {0, 2}, {8, 2});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0, 2}, {8, 2});
    v2AndV3ReplicaGroupsMatch(
        v2_group_list, v3_group_list,
        /*expected_num_replica_groups=*/(16 * 8 * 16) / (8 * 2),
        /*expected_num_devices_per_group=*/8 * 2);
  }
}

TEST(SPMDPartitionerUtilTest,
     ValidateEqualityOfV2AndV3PartitionGroupsForReplication) {
  auto v2AndV3ReplicaGroupsMatch =
      [](std::optional<IotaReplicaGroupList> v2_group_list,
         std::optional<MeshAxesReplicaGroupList> v3_group_list,
         int64_t expected_num_replica_groups,
         int64_t expected_num_devices_per_group) {
        EXPECT_TRUE(v2_group_list.has_value());
        EXPECT_TRUE(v3_group_list.has_value());
        EXPECT_EQ(v2_group_list->num_replica_groups(),
                  expected_num_replica_groups);
        EXPECT_EQ(v3_group_list->num_replica_groups(),
                  expected_num_replica_groups);
        EXPECT_EQ(v2_group_list->num_devices_per_group(),
                  expected_num_devices_per_group);
        EXPECT_EQ(v3_group_list->num_devices_per_group(),
                  expected_num_devices_per_group);
        EXPECT_EQ(v2_group_list->flattened_replica_groups(),
                  v3_group_list->flattened_replica_groups());
      };
  {
    HloSharding sharding = HloSharding::IotaTile({32, 32});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsForReplication(sharding, {0});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsForReplication(sharding, {0});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/32,
                              /*expected_num_devices_per_group=*/32);
  }
  {
    HloSharding sharding = HloSharding::IotaTile({16, 64});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsForReplication(sharding, {1});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsForReplication(sharding, {1});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/16,
                              /*expected_num_devices_per_group=*/64);
  }
  {
    HloSharding sharding = HloSharding::IotaTile({8, 8, 16});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsForReplication(sharding, {0, 1});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsForReplication(sharding, {0, 1});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/16,
                              /*expected_num_devices_per_group=*/64);
  }
  {
    HloSharding sharding = HloSharding::IotaTile({16, 8, 16});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsForReplication(sharding, {0, 2});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsForReplication(sharding, {0, 2});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/8,
                              /*expected_num_devices_per_group=*/256);
  }
}

}  // namespace
}  // namespace spmd
}  // namespace xla
