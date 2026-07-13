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

#include "xla/backends/gpu/collectives/gpu_clique_key.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/btree_map.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

static GpuCliqueKey GetBaseCliqueKey() {
  return GpuCliqueKey(/*devices=*/{GlobalDeviceId(0), GlobalDeviceId(1)},
                      /*num_local_participants=*/2);
}

TEST(GpuCliqueKeyTest, IsSubsetOf) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  GpuCliqueKey key0({id0, id1}, /*num_local_participants=*/2);
  GpuCliqueKey key1({id0, id1, id2, id3}, /*num_local_participants=*/4);
  GpuCliqueKey key2({id0, id1, id2, id3}, /*num_local_participants=*/4,
                    CommunicationId(1));
  GpuCliqueKey key3({id1, id2, id3}, /*num_local_participants=*/3);

  EXPECT_TRUE(key0.IsSubsetOf(key1));
  EXPECT_FALSE(key0.IsSubsetOf(key2));
  EXPECT_FALSE(key0.IsSubsetOf(key3));
}

TEST(GpuCliqueKeyTest, GetRootDevices) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  GpuCliqueKey key({id0, id1, id2, id3}, /*num_local_participants=*/1);

  {
    std::vector<GlobalDeviceId> roots = key.GetRootDevices(1);
    std::vector<GlobalDeviceId> expected = {id0};
    ASSERT_EQ(roots, expected);
  }

  {
    std::vector<GlobalDeviceId> roots = key.GetRootDevices(2);
    std::vector<GlobalDeviceId> expected = {id0, id2};
    ASSERT_EQ(roots, expected);
  }

  {
    std::vector<GlobalDeviceId> roots = key.GetRootDevices(3);
    std::vector<GlobalDeviceId> expected = {id0, id2, id3};
    ASSERT_EQ(roots, expected);
  }

  {
    std::vector<GlobalDeviceId> roots = key.GetRootDevices(4);
    std::vector<GlobalDeviceId> expected = {id0, id1, id2, id3};
    ASSERT_EQ(roots, expected);
  }
}

TEST(GpuCliqueKeyTest, Compare) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  GpuCliqueKey key0({id0, id1}, /*num_local_participants=*/1);
  GpuCliqueKey key1({id1, id2, id3}, /*num_local_participants=*/1);

  EXPECT_LT(key0, key1);
  EXPECT_GT(key1, key0);
}

TEST(GpuCliqueKeyTest, BtreeIterationOrder) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  GpuCliqueKey key0({id0, id2}, /*num_local_participants=*/1);
  GpuCliqueKey key1({id0, id1, id2, id3},
                    /*num_local_participants=*/1);

  absl::btree_map<GpuCliqueKey, int64_t, std::greater<GpuCliqueKey>> map;
  map[key0] = 0;
  map[key1] = 1;

  EXPECT_EQ(map.begin()->first, key1);
}

TEST(GpuCliqueKeyGettersTest, Devices) {
  EXPECT_THAT(
      GetBaseCliqueKey().devices(),
      ::testing::UnorderedElementsAre(GlobalDeviceId(0), GlobalDeviceId(1)));
}

TEST(GpuCliqueKeyGettersTest, Rank) {
  auto key = GetBaseCliqueKey();
  EXPECT_EQ(key.rank(GlobalDeviceId(0)), 0);
  EXPECT_EQ(key.rank(GlobalDeviceId(1)), 1);
  EXPECT_EQ(key.rank(GlobalDeviceId(2)), std::nullopt);
  EXPECT_EQ(key.rank(GlobalDeviceId(3)), std::nullopt);
}

TEST(GpuCliqueKeyGettersTest, CommunicationId) {
  EXPECT_EQ(GetBaseCliqueKey().communication_id(), CommunicationId(0));
}

TEST(GpuCliqueKeyGetterTest, ToString) {
  EXPECT_EQ(GetBaseCliqueKey().ToString(),
            "devices=2:[0,1]; local_participants=2; communication_id=0; "
            "incarnations=[]");
}

TEST(GpuCliqueKeyGetterTest, ToStringManyDevices) {
  std::vector<GlobalDeviceId> devices;
  devices.reserve(100);
  for (size_t i = 0; i < 100; ++i) {
    devices.push_back(GlobalDeviceId(i));
  }

  GpuCliqueKey key(devices, 100);

  EXPECT_EQ(key.ToString(),
            "devices=100:[0,1,2,3,4,5,6,7...98,99]; local_participants=100; "
            "communication_id=0; incarnations=[]");
}

TEST(GpuCliqueIdGettersTest, Data) {
  std::array<char, 129> id;
  std::fill(id.begin(), id.end(), 0x01);
  id[128] = 0;
  CliqueId clique_id(id.data());
  EXPECT_EQ(std::memcmp(clique_id.data().data(), id.data(), 128), 0);
}

TEST(GpuCliqueIdStringTest, ToString) {
  std::array<char, 129> id;
  std::fill(id.begin(), id.end(), 0x01);
  id[128] = 0;
  CliqueId clique_id(id.data());
  for (int i = 0; i < 128; ++i) {
    EXPECT_EQ(clique_id.ToString()[i], id[i]);
  }
}

// Test that IsSubsetOf correctly identifies subset relationships for clique
// invalidation. This is important for preventing deadlocks when cached subset
// cliques need to be invalidated after a larger clique is created.
TEST(GpuCliqueKeyTest, IsSubsetOfForCliqueInvalidation) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  // Scenario: First we create clique [0,1], then later create [0,1,2,3].
  // The cached [0,1] clique should be identified as a subset of [0,1,2,3]
  // so it can be invalidated to prevent deadlock during comm splitting.
  GpuCliqueKey small_clique({id0, id1}, /*num_local_participants=*/2);
  GpuCliqueKey large_clique({id0, id1, id2, id3},
                            /*num_local_participants=*/4);

  // [0,1] is a subset of [0,1,2,3]
  EXPECT_TRUE(small_clique.IsSubsetOf(large_clique));

  // [0,1,2,3] is not a subset of [0,1]
  EXPECT_FALSE(large_clique.IsSubsetOf(small_clique));

  // A clique is a subset of itself
  EXPECT_TRUE(small_clique.IsSubsetOf(small_clique));
  EXPECT_TRUE(large_clique.IsSubsetOf(large_clique));

  // [2,3] is also a subset of [0,1,2,3]
  GpuCliqueKey other_small({id2, id3}, /*num_local_participants=*/2);
  EXPECT_TRUE(other_small.IsSubsetOf(large_clique));

  // [0,1] is not a subset of [2,3] (disjoint)
  EXPECT_FALSE(small_clique.IsSubsetOf(other_small));

  // Different communication_id should prevent subset relationship
  GpuCliqueKey diff_comm_clique({id0, id1}, /*num_local_participants=*/2,
                                CommunicationId(1));
  EXPECT_FALSE(diff_comm_clique.IsSubsetOf(large_clique));
}

// Test that IsSubsetOf correctly compares incarnations to prevent false
// positives. This is a regression test for a bug where incarnations were
// not being compared, causing cliques with different incarnations to be
// incorrectly identified as subsets, leading to deadlocks.
TEST(GpuCliqueKeyTest, IsSubsetOfComparesIncarnations) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  // Create two cliques with the same devices and communication_id,
  // but different incarnations
  std::vector<IncarnationId> incarnations1 = {IncarnationId(1),
                                              IncarnationId(2)};
  std::vector<IncarnationId> incarnations2 = {IncarnationId(3),
                                              IncarnationId(4)};

  GpuCliqueKey key_with_incarnations1({id0, id1}, /*num_local_participants=*/2,
                                      CommunicationId(0), incarnations1);
  GpuCliqueKey key_with_incarnations2({id0, id1}, /*num_local_participants=*/2,
                                      CommunicationId(0), incarnations2);

  // These keys should NOT be subsets of each other because their
  // incarnations differ, even though devices and communication_id match
  EXPECT_FALSE(key_with_incarnations1.IsSubsetOf(key_with_incarnations2));
  EXPECT_FALSE(key_with_incarnations2.IsSubsetOf(key_with_incarnations1));

  // Test subset relationship with matching incarnations
  GpuCliqueKey small_key({id0, id1}, /*num_local_participants=*/2,
                         CommunicationId(0), incarnations1);
  GpuCliqueKey large_key({id0, id1, id2, id3}, /*num_local_participants=*/4,
                         CommunicationId(0), incarnations1);

  // With matching incarnations, the subset relationship should work
  EXPECT_TRUE(small_key.IsSubsetOf(large_key));
  EXPECT_FALSE(large_key.IsSubsetOf(small_key));

  // But with different incarnations, it should fail even if devices are a
  // subset
  GpuCliqueKey small_key_different_incarnations(
      {id0, id1}, /*num_local_participants=*/2, CommunicationId(0),
      incarnations2);

  EXPECT_FALSE(small_key_different_incarnations.IsSubsetOf(large_key));

  // Keys with same devices and incarnations should be subsets of themselves
  EXPECT_TRUE(key_with_incarnations1.IsSubsetOf(key_with_incarnations1));
  EXPECT_TRUE(key_with_incarnations2.IsSubsetOf(key_with_incarnations2));
}

TEST(GpuCliqueKeyTest, EqualityIncludesCommunicationId) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);

  GpuCliqueKey key0({id0, id1}, /*num_local_participants=*/2,
                    CommunicationId(0));
  GpuCliqueKey key1({id0, id1}, /*num_local_participants=*/2,
                    CommunicationId(1));
  GpuCliqueKey key2({id0, id1}, /*num_local_participants=*/2,
                    CommunicationId(0));

  EXPECT_NE(key0, key1);
  EXPECT_EQ(key0, key2);
}

TEST(GpuCliqueKeyTest, CompareIncludesCommunicationId) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);

  GpuCliqueKey key0({id0, id1}, /*num_local_participants=*/2,
                    CommunicationId(0));
  GpuCliqueKey key1({id0, id1}, /*num_local_participants=*/2,
                    CommunicationId(1));

  EXPECT_LT(key0, key1);
  EXPECT_GT(key1, key0);
  EXPECT_FALSE(key0 > key1);
  EXPECT_FALSE(key1 < key0);
}

TEST(GpuCliqueKeyTest, BtreeDistinguishesCommunicationId) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);

  GpuCliqueKey key0({id0, id1}, /*num_local_participants=*/2,
                    CommunicationId(0));
  GpuCliqueKey key1({id0, id1}, /*num_local_participants=*/2,
                    CommunicationId(1));

  absl::btree_map<GpuCliqueKey, int64_t, std::less<GpuCliqueKey>> map;
  map[key0] = 0;
  map[key1] = 1;

  // Both keys should be present as separate entries.
  EXPECT_EQ(map.size(), 2);
  EXPECT_EQ(map[key0], 0);
  EXPECT_EQ(map[key1], 1);
}

}  // namespace xla::gpu
