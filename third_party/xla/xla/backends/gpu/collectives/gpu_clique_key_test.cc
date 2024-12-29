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
#include "xla/service/global_device_id.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

static GpuCliqueKey GetBaseCliqueKey() {
  return GpuCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1)},
                      CollectiveStreamId(0), AsyncStreamKind::kCollective,
                      std::vector<std::vector<GlobalDeviceId>>{
                          {GlobalDeviceId(0), GlobalDeviceId(1)},
                          {GlobalDeviceId(2), GlobalDeviceId(3)}});
}
TEST(GpuCliqueKeyTest, IsSubsetOf) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  GpuCliqueKey key0({id0, id1}, CollectiveStreamId(0));
  GpuCliqueKey key1({id0, id1, id2, id3}, CollectiveStreamId(0));
  GpuCliqueKey key2({id0, id1, id2, id3}, CollectiveStreamId(1));
  GpuCliqueKey key3({id1, id2, id3}, CollectiveStreamId(0));

  EXPECT_TRUE(key0.IsSubsetOf(key1));
  EXPECT_FALSE(key0.IsSubsetOf(key2));
  EXPECT_FALSE(key0.IsSubsetOf(key3));
}

TEST(GpuCliqueKeyTest, Compare) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  GpuCliqueKey key0({id0, id1}, CollectiveStreamId(0));
  GpuCliqueKey key1({id1, id2, id3}, CollectiveStreamId(0));
  GpuCliqueKey key2({id1, id2, id3}, CollectiveStreamId(1));

  EXPECT_LT(key0, key1);
  EXPECT_GT(key1, key0);
  EXPECT_LT(key1, key2);
}

TEST(GpuCliqueKeyTest, CompareWithParticipantGroups) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  // The keys are not equal because the replica groups are different.
  GpuCliqueKey key0({id0, id1}, CollectiveStreamId(0),
                    AsyncStreamKind::kCollective,
                    std::vector<std::vector<GlobalDeviceId>>{{id0, id1}});
  GpuCliqueKey key1(
      {id0, id1}, CollectiveStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id0, id1}, {id2, id3}});
  EXPECT_FALSE(key0 == key1);

  // With no replica groups, the keys are equal
  GpuCliqueKey key0_nogroups({id0, id1}, CollectiveStreamId(0));
  GpuCliqueKey key1_nogroups({id0, id1}, CollectiveStreamId(0));
  EXPECT_EQ(key0_nogroups, key1_nogroups);
}

TEST(GpuCliqueKeyTest, CompareWithPermutedParticipantGroups) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  // The keys are equal because the replica groups are same up to permutation.
  GpuCliqueKey key0(
      {id0, id1}, CollectiveStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id3, id2}, {id0, id1}});
  GpuCliqueKey key1(
      {id0, id1}, CollectiveStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id0, id1}, {id2, id3}});
  EXPECT_EQ(key0, key1);

  GpuCliqueKey key_other(
      {id0, id1}, CollectiveStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id0, id2}, {id1, id3}});
  EXPECT_FALSE(key0 == key_other);
}

TEST(GpuCliqueKeyTest, BtreeIterationOrder) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  GpuCliqueKey key0({id0, id2}, CollectiveStreamId(0));
  GpuCliqueKey key1({id0, id1, id2, id3}, CollectiveStreamId(0));

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

TEST(GpuCliqueKeyGettersTest, StreamId) {
  EXPECT_EQ(GetBaseCliqueKey().stream_id(), CollectiveStreamId(0));
}

TEST(GpuCliqueKeyGetterTest, ToString) {
  EXPECT_EQ(GetBaseCliqueKey().ToString(),
            "devices=[0,1]; stream=0; groups=[[0,1],[2,3]]");
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

}  // namespace xla::gpu
