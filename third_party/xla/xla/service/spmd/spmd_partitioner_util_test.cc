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

#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_sharding.h"
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

}  // namespace
}  // namespace spmd
}  // namespace xla
