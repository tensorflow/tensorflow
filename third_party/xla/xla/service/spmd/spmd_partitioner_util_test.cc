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
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/array.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/mesh_and_axis.h"
#include "xla/hlo/ir/named_sharding.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/service/spmd/spmd_partitioner_util_internal.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

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

  {
    Mesh mesh({2, 2}, {"a", "b"});
    HloSharding partial_sharding(test_utils::FromAxisNames(mesh, {{}, {"a"}}));
    const std::vector<HloSharding> target_shardings = {
        HloSharding(test_utils::FromAxisNames(mesh, {{"a"}, {"b"}})),
        HloSharding(test_utils::FromAxisNames(mesh, {{"b"}, {"a"}}))};

    for (const auto& target_sharding : target_shardings) {
      auto result = PartialReplicateReshardCompatibleSharding(partial_sharding,
                                                              target_sharding);

      EXPECT_EQ(result, target_shardings[1]);
    }

    partial_sharding =
        HloSharding(test_utils::FromAxisNames(mesh, {{}, {"b"}}));
    for (const auto& target_sharding : target_shardings) {
      auto result = PartialReplicateReshardCompatibleSharding(partial_sharding,
                                                              target_sharding);

      EXPECT_EQ(result, target_shardings[0]);
    }
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

  {
    Mesh mesh({2, 2, 2, 2, 2}, {"a", "b", "c", "d", "e"});
    HloSharding v3_partial_sharding(
        test_utils::FromAxisNames(mesh, {{"a"}, {"b"}}));
    const std::vector<HloSharding> v3_target_shardings = {
        HloSharding(test_utils::FromAxisNames(mesh, {{"a", "c"}, {"b", "d"}})),
        HloSharding(test_utils::FromAxisNames(mesh, {{"a", "c"}, {"b", "e"}})),
        HloSharding(test_utils::FromAxisNames(mesh, {{"a", "d"}, {"b", "c"}})),
        HloSharding(test_utils::FromAxisNames(mesh, {{"a", "d"}, {"b", "e"}})),
        HloSharding(test_utils::FromAxisNames(mesh, {{"a", "e"}, {"b", "c"}})),
        HloSharding(test_utils::FromAxisNames(mesh, {{"a", "e"}, {"b", "d"}}))};

    for (const auto& target_sharding : v3_target_shardings) {
      auto result = PartialReplicateReshardCompatibleSharding(
          v3_partial_sharding, target_sharding);

      EXPECT_EQ(result, target_sharding);
    }
  }
}

TEST(SPMDPartitionerUtilTest, PartialReplicateReshardCompatibleShardingV3) {
  // Perfect matching axis (no sub-axis splitting necessary).
  Mesh mesh({2, 2}, {"a", "b"});

  std::optional<HloSharding> result = PartialReplicateReshardCompatibleSharding(
      HloSharding(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"b"}})),
      HloSharding(
          test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a", "b"}})));

  ASSERT_TRUE(result.has_value());
  // Expanding by 2 automatically absorbs the full "a" axis (size 2) from
  // implicit replicated axes.
  EXPECT_EQ(result->named_sharding(),
            test_utils::FromAxisNames(mesh,
                                      /*dim_shardings=*/{{"b", "a"}}));
}

TEST(SPMDPartitionerUtilTest,
     PartialReplicateReshardIncompatibleShardingV3ExplicitReplicationConflict) {
  Mesh mesh({4, 2}, {"x", "y"});

  // Since partial sharding explicitly requires "y" to be a data dimension,
  // these shardings are incompatible.
  std::optional<HloSharding> result = PartialReplicateReshardCompatibleSharding(
      HloSharding(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"y"}})),
      HloSharding(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"x"}},
                                            /*replicated_axes=*/{"y"})));

  EXPECT_FALSE(result.has_value());
}

TEST(SPMDPartitionerUtilTest,
     PartialReplicateReshardCompatibleShardingV3SubAxes) {
  Mesh mesh({4, 2}, {"x", "y"});

  // This means the reshard expands dim0 from size 2 to size 4.
  // It needs 2 replication devices from the available implicit replicated axis
  // "x" (which has size 4).
  // Target does not explicitly request "y" to be replicated, so the sub-axis
  // expansion is fully compatible.
  std::optional<HloSharding> result = PartialReplicateReshardCompatibleSharding(
      HloSharding(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"y"}})),
      HloSharding(test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"x"}})));

  ASSERT_TRUE(result.has_value());
  // The computed compatible sharding leaves the unneeded sub-axis of "x" as
  // implicit and assigns a sub-axis chunk of "x" to the data-dimension.
  EXPECT_EQ(result->named_sharding(),
            test_utils::FromAxisNames(mesh,
                                      /*dim_shardings=*/{{"y", "x:(1)2"}}));
}

TEST(SPMDPartitionerUtilTest, GetListOfListsPartitionGroupsForReplication) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2});
  CollectiveDeviceList actual_partition_groups =
      GetListOfListsPartitionGroupsForReplication(sharding, {1});
  std::vector<std::vector<int64_t>> expected_partition_groups = {
      {0, 2}, {1, 3}, {4, 6}, {5, 7}};
  EXPECT_THAT(actual_partition_groups.flattened_replica_groups(),
              testing::ContainerEq(expected_partition_groups));
}

TEST(SPMDPartitionerUtilTest, GetListOfListsPartitionGroupsForReplication2) {
  HloSharding sharding = HloSharding::IotaTile({2, 2, 2}, {2, 2, 2}, {0, 2, 1});
  CollectiveDeviceList actual_partition_groups =
      GetListOfListsPartitionGroupsForReplication(sharding, {0, 2});
  std::vector<std::vector<int64_t>> expected_partition_groups = {{0, 2, 4, 6},
                                                                 {1, 3, 5, 7}};
  EXPECT_THAT(actual_partition_groups.flattened_replica_groups(),
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
      ExpandPartitionGroupListAcrossReplicas(partition_group_list, 2, 50);
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
  EXPECT_EQ(
      v3_group_list->ToString(),
      "mesh['axis_0'=8,'axis_1'=8,'axis_2'=16] {'axis_0':(2)4,'axis_1':(2)4}");

  // V3 Sharding (Will correctly reflect the real mesh axis names)
  Mesh mesh({8, 8, 16}, {"a", "b", "c"});
  NamedSharding named_sharding =
      test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}, {"b"}, {"c"}});
  HloSharding sharding_v3 = HloSharding(named_sharding);
  v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding_v3, {0, 1}, {4, 4});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 64);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 16);
  EXPECT_EQ(v3_group_list->ToString(),
            "mesh['a'=8,'b'=8,'c'=16] {'a':(2)4,'b':(2)4}");
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsAcrossTargetDimsV3GroupSizeEqAxisProduct) {
  Mesh mesh({4, 2, 3}, {"a", "b", "c"});
  // In this non-positional sharding:
  // - Dimension 0 is sharded across mesh axes "a" and "b" (sharding size 8).
  // - Dimension 1 is sharded across mesh axis "c" (sharding size 3).
  NamedSharding named_sharding =
      test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a", "b"}, {"c"}});
  HloSharding sharding(named_sharding);
  // We expect to use all mesh axes associated with dim0, which are "a" and "b".
  std::optional<MeshAxesReplicaGroupList> v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0}, {8});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 3);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 8);
  EXPECT_EQ(v3_group_list->ToString(), "mesh['a'=4,'b'=2,'c'=3] {'a','b'}");
  // Group across dimension 1 with group size 3.
  // We expect to take the mesh axis associated with dim1, which is "c".
  v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {1}, {3});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 8);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 3);
  EXPECT_EQ(v3_group_list->ToString(), "mesh['a'=4,'b'=2,'c'=3] {'c'}");
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsAcrossTargetDimsV3GroupSizeNeqAxisProduct) {
  Mesh mesh({4, 2, 3}, {"a", "b", "c"});
  // In this non-positional sharding:
  // - Dimension 0 is sharded across mesh axes "a" and "b" (sharding size 8).
  // - Dimension 1 is sharded across mesh axis "c" (sharding size 3).
  NamedSharding named_sharding =
      test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a", "b"}, {"c"}});
  HloSharding sharding(named_sharding);
  // The replica group will fully consume the minor most axis, "b", but we need
  // to take half of the major most axis, "a" to get the desired group size. We
  // use a sub-axis of "a" which fits the remaining desired group size.
  std::optional<MeshAxesReplicaGroupList> v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0}, {4});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 6);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 4);
  EXPECT_EQ(v3_group_list->ToString(),
            "mesh['a'=4,'b'=2,'c'=3] {'a':(2)2,'b'}");
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsAcrossTargetDimsSubAxesNeedsSplitting) {
  Mesh mesh({8, 4}, {"a", "b"});
  // dim0 is sharded over a 16-shard grid using a sub-portion of axis "a" and
  // the full axis "b".
  // Specifically:
  // - "a" (axis 0) is split into [pre=2, size=4, post=1].
  // - "b" (axis 1) is taken in full (size 4).
  NamedSharding named_sharding =
      test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a:(2)4", "b"}});
  HloSharding sharding(named_sharding);

  // Our group size is 2. Since "b" is the most minor axis and has size 4, we
  // split it into two groups of 2.
  // The result should be the minor half of "b".
  std::optional<MeshAxesReplicaGroupList> v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0}, {2});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 16);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 2);
  EXPECT_EQ(v3_group_list->ToString(), "mesh['a'=8,'b'=4] {'b':(2)2}");
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsAcrossTargetDimsSubAxesNoSplitting) {
  Mesh mesh({8, 4}, {"a", "b"});
  // dim0 is sharded over a 16-shard grid using a sub-portion of axis "a" and
  // the full axis "b".
  // Specifically:
  // - "a" (axis 0) is split into [pre=2, size=4, post=1].
  // - "b" (axis 1) is taken in full (size 4).
  NamedSharding named_sharding =
      test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a:(2)4", "b"}});
  HloSharding sharding(named_sharding);

  // For this group size (8) we consume the full minor axis of the dimension
  // ("b") which has size 4. Then we also ned a minor axis from "a" of size 2.
  // Since the sub-axis of "a" was already halved (pre_size 2, size 4), we slice
  // it again to get the minor axis for the replica group. This is the sub-axis
  // of "a" with pre_size 2 * (4/2) = 4 and size 2.
  std::optional<MeshAxesReplicaGroupList> v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0}, {8});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 4);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 8);
  EXPECT_EQ(v3_group_list->ToString(), "mesh['a'=8,'b'=4] {'a':(4)2,'b'}");

  // Now reverse the order of the mesh axes in the dim sharding. The a sub-axis
  // should not need splitting, since it is now minormost. The b axis is now
  // the major axis and will be split.
  named_sharding =
      test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"b", "a:(2)4"}});
  sharding = HloSharding(named_sharding);
  v3_group_list =
      GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0}, {8});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 4);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 8);
  EXPECT_EQ(v3_group_list->ToString(), "mesh['a'=8,'b'=4] {'b':(2)2,'a':(2)4}");
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
            "mesh['axis_0'=2,'axis_1'=2,'axis_2'=2] {'axis_1'}");

  // V3 Sharding
  HloSharding sharding_v3(test_utils::FromAxisNames(
      Mesh({2, 2, 2}, {"Q", "K", "V"}), {{"Q"}, {"K"}, {"V"}}));
  v3_group_list = GetMeshAxesPartitionGroupsForReplication(sharding_v3, {1});
  EXPECT_TRUE(v3_group_list.has_value());
  EXPECT_EQ(v3_group_list->num_replica_groups(), 4);
  EXPECT_EQ(v3_group_list->num_devices_per_group(), 2);
  EXPECT_EQ(v3_group_list->ToString(), "mesh['Q'=2,'K'=2,'V'=2] {'K'}");
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsForReplicationNamedShardingNonPositional) {
  HloSharding sharding(
      test_utils::FromAxisNames(Mesh({2, 2}, {"a", "b"}),
                                /*dim_shardings=*/{{"b"}, {"a"}}));

  // Replicate across dim 0 then dim 1. Expect axes order {"b", "a"}.
  std::optional<MeshAxesReplicaGroupList> groups1 =
      GetMeshAxesPartitionGroupsForReplication(sharding, {0, 1});
  EXPECT_TRUE(groups1.has_value());
  EXPECT_EQ(groups1->ToString(), "mesh['a'=2,'b'=2] {'b','a'}");

  // Replicate across dim 1 then dim 0. Expect axes order {"a", "b"}.
  std::optional<MeshAxesReplicaGroupList> groups2 =
      GetMeshAxesPartitionGroupsForReplication(sharding, {1, 0});
  EXPECT_TRUE(groups2.has_value());
  EXPECT_EQ(groups2->ToString(), "mesh['a'=2,'b'=2] {'a','b'}");
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsForReplicationNamedShardingMergeSubAxes) {
  Mesh mesh({4}, {"a"});
  NamedSharding named_sharding = test_utils::FromAxisNames(
      mesh, /*dim_shardings=*/{{"a:(1)2"}, {"a:(2)2"}});
  HloSharding sharding(named_sharding);

  // When replicating over {0, 1}, the contiguous sub-axes a:(1)2 and a:(2)2 are
  // merged into a.
  std::optional<MeshAxesReplicaGroupList> groups =
      GetMeshAxesPartitionGroupsForReplication(sharding, {0, 1});
  EXPECT_TRUE(groups.has_value());
  EXPECT_EQ(groups->ToString(), "mesh['a'=4] {'a'}");
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsForReplicationNamedShardingHigherTensorRank) {
  HloSharding sharding(test_utils::FromAxisNames(
      Mesh({2}, {"a"}), /*dim_shardings=*/{{}, {"a"}}));

  // Replication is requested for dimension 1, which is sharded on axis "a".
  std::optional<MeshAxesReplicaGroupList> groups =
      GetMeshAxesPartitionGroupsForReplication(sharding, {1});

  EXPECT_TRUE(groups.has_value());
  EXPECT_EQ(groups->ToString(), "mesh['a'=2] {'a'}");
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
  {
    // Non-trivial v2 sharding.
    HloSharding sharding =
        HloSharding::IotaTile({6, 35}, {7, 10, 3}, {2, 1, 0});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsAcrossTargetDims(sharding, {0, 1}, {3, 5});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsAcrossTargetDims(sharding, {0, 1}, {3, 5});
    // V2 representation cannot create a compressed representation...
    EXPECT_FALSE(v2_group_list.has_value());
    // ... but V3 can via subaxes.
    EXPECT_TRUE(v3_group_list.has_value());
    EXPECT_EQ(v3_group_list->num_replica_groups(), (6 * 35) / (3 * 5));
    EXPECT_EQ(v3_group_list->num_devices_per_group(), 3 * 5);
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
  // Test non-trivial v2 shardings.
  {
    HloSharding sharding =
        HloSharding::IotaTile({6, 35}, {7, 10, 3}, {2, 1, 0});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsForReplication(sharding, {0});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsForReplication(sharding, {0});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/35,
                              /*expected_num_devices_per_group=*/6);
    EXPECT_EQ(v2_group_list->ToString(), "[35,6]<=[7,2,5,3]T(2,0,3,1)");
  }
  {
    HloSharding sharding =
        HloSharding::IotaTile({6, 35}, {7, 10, 3}, {2, 1, 0});
    std::optional<IotaReplicaGroupList> v2_group_list =
        GetIotaPartitionGroupsForReplication(sharding, {1});
    std::optional<MeshAxesReplicaGroupList> v3_group_list =
        GetMeshAxesPartitionGroupsForReplication(sharding, {1});
    v2AndV3ReplicaGroupsMatch(v2_group_list, v3_group_list,
                              /*expected_num_replica_groups=*/6,
                              /*expected_num_devices_per_group=*/35);
    EXPECT_EQ(v2_group_list->ToString(), "[6,35]<=[7,10,3]T(2,1,0)");
  }
}

TEST(SPMDPartitionerUtilTest,
     GetMeshAxesPartitionGroupsForReplicationV3OrderMatters) {
  Mesh mesh({2, 2, 2}, {"a", "b", "c"});
  NamedSharding named_sharding =
      test_utils::FromAxisNames(mesh, /*dim_shardings=*/{{"a"}, {"b"}, {"c"}});
  HloSharding sharding(named_sharding);

  // Replicating across dim 0 then dim 1.
  std::optional<MeshAxesReplicaGroupList> v3_group_list_1 =
      GetMeshAxesPartitionGroupsForReplication(sharding, {0, 1});

  // Replicating across dim 1 then dim 0.
  std::optional<MeshAxesReplicaGroupList> v3_group_list_2 =
      GetMeshAxesPartitionGroupsForReplication(sharding, {1, 0});

  EXPECT_TRUE(v3_group_list_1.has_value());
  EXPECT_TRUE(v3_group_list_2.has_value());

  // They should NOT be equal because order of replication dims matters.
  EXPECT_NE(v3_group_list_1->flattened_replica_groups(),
            v3_group_list_2->flattened_replica_groups());

  auto groups1 = v3_group_list_1->flattened_replica_groups();
  auto groups2 = v3_group_list_2->flattened_replica_groups();

  ASSERT_EQ(groups1.size(), groups2.size());
  for (int64_t i = 0; i < groups1.size(); ++i) {
    EXPECT_THAT(groups1[i], testing::UnorderedElementsAreArray(groups2[i]));
    EXPECT_NE(groups1[i], groups2[i]);
  }
}

TEST(SPMDPartitionerUtilTest, CanonicalizeShardingV2Transposed) {
  // Sharding: [2,4]<=[4,2]T(1,0)
  HloSharding sharding = HloSharding::IotaTile({2, 4}, {4, 2}, {1, 0});

  // For a sharding like [2,4]<=[4,2]T(1,0), we expect translation to a
  // canonical V3 representation with mesh=['axis_0'=4, 'axis_1'=2] and
  // sharding=[{"axis_1"}, {"axis_0"}].
  HloSharding canonicalized = xla::spmd::CanonicalizeSharding(sharding);

  EXPECT_TRUE(canonicalized.UseNamedShardingLeaf());
  EXPECT_EQ(canonicalized.named_sharding().ToString(),
            "{mesh['axis_0'=4,'axis_1'=2], [{'axis_1'}, {'axis_0'}]}");
}

TEST(SPMDPartitionerUtilTest, CanonicalizeShardingV2Trivial) {
  HloSharding sharding = HloSharding::IotaTile({2, 2});

  // Trivial iota should now be translated by HloSharding::ToNamedSharding.
  HloSharding canonicalized = xla::spmd::CanonicalizeSharding(sharding);

  EXPECT_TRUE(canonicalized.UseNamedShardingLeaf());
  EXPECT_EQ(canonicalized.named_sharding().ToString(),
            "{mesh['axis_0'=2,'axis_1'=2], [{'axis_0'}, {'axis_1'}]}");
}

TEST(SPMDPartitionerUtilTest, CanonicalizeShardingV1Iota) {
  // Create a V1 sharding with iota order.
  Array<int64_t> device_assignment({2, 2});
  device_assignment.FillIota(0);
  HloSharding sharding = HloSharding::Tile(device_assignment);

  // CanonicalizeSharding calls ToNamedSharding which will now translate
  // the V1 sharding to a V3 sharding.
  HloSharding canonicalized = xla::spmd::CanonicalizeSharding(sharding);

  EXPECT_TRUE(canonicalized.UseNamedShardingLeaf());
  EXPECT_EQ(canonicalized.named_sharding().ToString(),
            "{mesh['axis_0'=2,'axis_1'=2], [{'axis_0'}, {'axis_1'}]}");
}

TEST(SPMDPartitionerUtilTest, CanonicalizeShardingV1NonIota) {
  // Create a V1 sharding with non-iota order.
  Array<int64_t> device_assignment({2, 2});
  device_assignment(0, 0) = 0;
  device_assignment(0, 1) = 2;
  device_assignment(1, 0) = 1;
  device_assignment(1, 1) = 3;
  HloSharding sharding = HloSharding::Tile(device_assignment);

  // Non-iota shardings are also translated using the physical device order.
  HloSharding canonicalized = xla::spmd::CanonicalizeSharding(sharding);

  EXPECT_TRUE(canonicalized.UseNamedShardingLeaf());
  EXPECT_EQ(canonicalized.named_sharding().ToString(),
            "{mesh['axis_0'=2,'axis_1'=2], device_ids=(0,2,1,3), [{'axis_0'}, "
            "{'axis_1'}]}");
}

TEST(SPMDPartitionerUtilTest, GetPartitionGroupsForReplicationGating) {
  HloSharding sharding = HloSharding::IotaTile({8, 8, 16});

  // With enable_rgv3 = true, should return kMeshAxes (V3) representation
  std::unique_ptr<CollectiveDeviceListBase> groups_v3 =
      GetPartitionGroupsForReplication(sharding, {0, 1}, /*enable_rgv3=*/true);
  EXPECT_EQ(groups_v3->version(), CollectiveDeviceListVersion::kMeshAxes);

  // With enable_rgv3 = false, should fall back to kIota or kListOfLists
  // representation
  std::unique_ptr<CollectiveDeviceListBase> groups_fallback =
      GetPartitionGroupsForReplication(sharding, {0, 1}, /*enable_rgv3=*/false);
  EXPECT_NE(groups_fallback->version(), CollectiveDeviceListVersion::kMeshAxes);
  EXPECT_EQ(groups_fallback->flattened_replica_groups(),
            groups_v3->flattened_replica_groups());
}

TEST(SPMDPartitionerUtilTest, GetPartitionGroupsAcrossTargetDimsGating) {
  HloSharding sharding = HloSharding::IotaTile({8, 8, 16});

  // With enable_rgv3 = true, should return kMeshAxes (V3) representation
  std::unique_ptr<CollectiveDeviceListBase> groups_v3 =
      GetPartitionGroupsAcrossTargetDims(sharding, {0, 1}, {4, 4},
                                         /*enable_rgv3=*/true);
  EXPECT_EQ(groups_v3->version(), CollectiveDeviceListVersion::kMeshAxes);

  // With enable_rgv3 = false, should fall back to kIota or kListOfLists
  // representation
  std::unique_ptr<CollectiveDeviceListBase> groups_fallback =
      GetPartitionGroupsAcrossTargetDims(sharding, {0, 1}, {4, 4},
                                         /*enable_rgv3=*/false);
  EXPECT_NE(groups_fallback->version(), CollectiveDeviceListVersion::kMeshAxes);
  EXPECT_EQ(groups_fallback->flattened_replica_groups(),
            groups_v3->flattened_replica_groups());
}

TEST(SPMDPartitionerUtilTest,
     CanReshardWithCollectivePermuteImplicitAndExplicitReplication) {
  Mesh mesh({2, 2, 2, 1}, {"d", "f", "e", "c"});
  HloSharding source =
      HloSharding(test_utils::FromAxisNames(mesh, {{"f"}, {}, {}}));
  HloSharding target =
      HloSharding(test_utils::FromAxisNames(mesh, {{"f"}, {}, {}},
                                            /*replicated_axes=*/{"d"}));

  // We don't want to collectively permute to a logically identical sharding as
  // this would be a copy / identity operation.
  EXPECT_FALSE(CanReshardWithCollectivePermute(source, target));
}

struct CanReshardWithCollectivePermuteTestCase {
  std::string name;
  HloSharding source;
  HloSharding target;
  bool expected_result;
};

class CanReshardWithCollectivePermuteEquivalenceTest
    : public ::testing::TestWithParam<CanReshardWithCollectivePermuteTestCase> {
};

TEST_P(CanReshardWithCollectivePermuteEquivalenceTest, Equivalence) {
  const auto& param = GetParam();
  const HloSharding& source_v2 = param.source;
  const HloSharding& target_v2 = param.target;
  HloSharding source_v3 = HloSharding::ToV3Sharding(source_v2);
  HloSharding target_v3 = HloSharding::ToV3Sharding(target_v2);

  bool result_v2 = CanReshardWithCollectivePermute(source_v2, target_v2);
  bool result_v3 = CanReshardWithCollectivePermute(source_v3, target_v3);

  EXPECT_EQ(result_v2, result_v3);
  EXPECT_EQ(result_v2, param.expected_result);
  // We also test that CanReshardWithCollectivePermute yields the same result
  // when mixing v2 and v3 shardings.
  EXPECT_EQ(CanReshardWithCollectivePermute(source_v2, target_v3), result_v2);
  EXPECT_EQ(CanReshardWithCollectivePermute(source_v3, target_v2), result_v2);
}

INSTANTIATE_TEST_SUITE_P(
    EquivalenceTests, CanReshardWithCollectivePermuteEquivalenceTest,
    testing::Values(
        CanReshardWithCollectivePermuteTestCase{
            "Replicated", HloSharding::Replicate(), HloSharding::Replicate(),
            false},
        CanReshardWithCollectivePermuteTestCase{
            "TileSame", HloSharding::IotaTile({2, 4}),
            HloSharding::IotaTile({2, 4}), false},
        CanReshardWithCollectivePermuteTestCase{
            "TileDiff", HloSharding::IotaTile({2, 4}),
            HloSharding::IotaTile({2, 4}, {4, 2}, {1, 0}), true},
        CanReshardWithCollectivePermuteTestCase{
            "TileDiffDims", HloSharding::IotaTile({2, 4}),
            HloSharding::IotaTile({4, 2}), false},
        CanReshardWithCollectivePermuteTestCase{
            "PartialTileSame",
            HloSharding::PartialTile(TileAssignment({1, 2, 4})),
            HloSharding::PartialTile(TileAssignment({1, 2, 4})), false},
        CanReshardWithCollectivePermuteTestCase{
            "PartialTileDiff",
            HloSharding::PartialTile(TileAssignment({1, 2, 4})),
            HloSharding::PartialTile(TileAssignment({1, 2, 4}, {4, 2}, {1, 0})),
            true},
        CanReshardWithCollectivePermuteTestCase{
            "SingleDevice", HloSharding::SingleDevice(0),
            HloSharding::SingleDevice(0), false},
        CanReshardWithCollectivePermuteTestCase{
            "SingleDeviceDiff", HloSharding::SingleDevice(0),
            HloSharding::SingleDevice(1), false}),
    [](const testing::TestParamInfo<CanReshardWithCollectivePermuteTestCase>&
           info) { return info.param.name; });

TEST(SPMDPartitionerUtilTest, FindRotateRightPatternOnRotate) {
  HloComputation::Builder builder("test");
  Shape shape = ShapeUtil::MakeShape(F32, {10, 20});
  auto* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto* rotate = builder.AddInstruction(
      HloInstruction::CreateRotate(shape, param, {0, 1}, {2, 5}));
  rotate->set_sharding(HloSharding::IotaTile({1, 2}));
  auto computation = builder.Build();
  rotate = computation->root_instruction();

  std::optional<RotateRightPatternMatch> match = FindRotateRightPattern(rotate);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->dim, 1);
  EXPECT_EQ(match->amount, 15);
  EXPECT_EQ(match->rotate_dim_idx, 1);
}

TEST(SPMDPartitionerUtilTest, FindRotateRightPatternOnConcat) {
  HloComputation::Builder builder("test");
  Shape shape = ShapeUtil::MakeShape(F32, {12});
  auto* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  param->set_sharding(HloSharding::IotaTile({4}));

  Shape slice_lhs_shape = ShapeUtil::MakeShape(F32, {2});
  auto* slice_lhs = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_lhs_shape, param, {10}, {12}, {1}));
  Shape slice_rhs_shape = ShapeUtil::MakeShape(F32, {10});
  auto* slice_rhs = builder.AddInstruction(
      HloInstruction::CreateSlice(slice_rhs_shape, param, {0}, {10}, {1}));

  auto* concat = builder.AddInstruction(
      HloInstruction::CreateConcatenate(shape, {slice_lhs, slice_rhs}, 0));
  concat->set_sharding(HloSharding::IotaTile({4}));
  auto computation = builder.Build();
  concat = computation->root_instruction();

  std::optional<RotateRightPatternMatch> match = FindRotateRightPattern(concat);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->dim, 0);
  EXPECT_EQ(match->amount, 2);
  EXPECT_EQ(match->rotate_dim_idx, -1);
}

TEST(SPMDPartitionerUtilTest, PopRotateDimension) {
  HloComputation::Builder builder("test");
  Shape shape = ShapeUtil::MakeShape(F32, {10, 20});
  auto* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto* rotate = builder.AddInstruction(
      HloInstruction::CreateRotate(shape, param, {0, 1}, {2, 5}));
  auto computation = builder.Build();
  rotate = computation->root_instruction();

  ASSERT_OK_AND_ASSIGN(HloInstruction * rem_rotate,
                       PopRotateDimension(rotate, /*rotate_dim_idx=*/1));
  EXPECT_EQ(rem_rotate, rotate);
  auto* rem_rotate_inst = Cast<HloRotateInstruction>(rem_rotate);
  ASSERT_EQ(rem_rotate_inst->dimensions().size(), 1);
  EXPECT_EQ(rem_rotate_inst->dimensions()[0], 0);
  EXPECT_EQ(rem_rotate_inst->shifts()[0], 2);

  ASSERT_OK_AND_ASSIGN(HloInstruction * final_inst,
                       PopRotateDimension(rem_rotate, /*rotate_dim_idx=*/0));
  EXPECT_EQ(final_inst, param);
}

TEST(SPMDPartitionerUtilTest, PopRotateDimensionMixedAndAllSharded) {
  HloComputation::Builder builder("test");
  Shape shape = ShapeUtil::MakeShape(F32, {8, 12, 16});
  auto* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0"));
  auto* rotate = builder.AddInstruction(
      HloInstruction::CreateRotate(shape, param, {0, 1, 2}, {1, 2, 3}));
  auto computation = builder.Build();
  rotate = computation->root_instruction();

  // 1. Pop sharded dimension 1 (`rotate_dim_idx = 1`). Replicated dims 0 and 2
  // stay in the rotate instruction.
  ASSERT_OK_AND_ASSIGN(HloInstruction * rem_rotate,
                       PopRotateDimension(rotate, /*rotate_dim_idx=*/1));
  EXPECT_EQ(rem_rotate, rotate);
  auto* rem_inst = Cast<HloRotateInstruction>(rem_rotate);
  ASSERT_EQ(rem_inst->dimensions().size(), 2);
  EXPECT_EQ(rem_inst->dimensions()[0], 0);
  EXPECT_EQ(rem_inst->dimensions()[1], 2);
  EXPECT_EQ(rem_inst->shifts()[0], 1);
  EXPECT_EQ(rem_inst->shifts()[1], 3);

  // 2. If all remaining dimensions (0 and 2) are subsequently popped, the
  // rotate instruction is completely replaced by its operand.
  ASSERT_OK_AND_ASSIGN(rem_rotate,
                       PopRotateDimension(rem_rotate, /*rotate_dim_idx=*/1));
  ASSERT_OK_AND_ASSIGN(HloInstruction * final_inst,
                       PopRotateDimension(rem_rotate, /*rotate_dim_idx=*/0));
  EXPECT_EQ(final_inst, param);
}

}  // namespace
}  // namespace spmd
}  // namespace xla
