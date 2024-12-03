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

#include "xla/service/gpu/runtime/nccl_clique_key.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "xla/service/global_device_id.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
using ::tsl::testing::StatusIs;
static NcclCliqueKey GetBaseCliqueKey() {
  return NcclCliqueKey({GlobalDeviceId(0), GlobalDeviceId(1)}, NcclStreamId(0),
                       AsyncStreamKind::kCollective,
                       std::vector<std::vector<GlobalDeviceId>>{
                           {GlobalDeviceId(0), GlobalDeviceId(1)},
                           {GlobalDeviceId(2), GlobalDeviceId(3)}});
}
TEST(NcclCliqueKeyTest, IsSubsetOf) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  NcclCliqueKey key0({id0, id1}, NcclStreamId(0));
  NcclCliqueKey key1({id0, id1, id2, id3}, NcclStreamId(0));
  NcclCliqueKey key2({id0, id1, id2, id3}, NcclStreamId(1));
  NcclCliqueKey key3({id1, id2, id3}, NcclStreamId(0));

  EXPECT_TRUE(key0.IsSubsetOf(key1));
  EXPECT_FALSE(key0.IsSubsetOf(key2));
  EXPECT_FALSE(key0.IsSubsetOf(key3));
}

TEST(NcclCliqueKeyTest, Compare) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  NcclCliqueKey key0({id0, id1}, NcclStreamId(0));
  NcclCliqueKey key1({id1, id2, id3}, NcclStreamId(0));
  NcclCliqueKey key2({id1, id2, id3}, NcclStreamId(1));

  EXPECT_LT(key0, key1);
  EXPECT_GT(key1, key0);
  EXPECT_LT(key1, key2);
}

TEST(NcclCliqueKeyTest, CompareWithParticipantGroups) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  // The keys are not equal because the replica groups are different.
  NcclCliqueKey key0({id0, id1}, NcclStreamId(0), AsyncStreamKind::kCollective,
                     std::vector<std::vector<GlobalDeviceId>>{{id0, id1}});
  NcclCliqueKey key1(
      {id0, id1}, NcclStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id0, id1}, {id2, id3}});
  EXPECT_FALSE(key0 == key1);

  // With no replica groups, the keys are equal
  NcclCliqueKey key0_nogroups({id0, id1}, NcclStreamId(0));
  NcclCliqueKey key1_nogroups({id0, id1}, NcclStreamId(0));
  EXPECT_EQ(key0_nogroups, key1_nogroups);
}

TEST(NcclCliqueKeyTest, CompareWithPermutedParticipantGroups) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  // The keys are equal because the replica groups are same up to permutation.
  NcclCliqueKey key0(
      {id0, id1}, NcclStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id3, id2}, {id0, id1}});
  NcclCliqueKey key1(
      {id0, id1}, NcclStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id0, id1}, {id2, id3}});
  EXPECT_EQ(key0, key1);

  NcclCliqueKey key_other(
      {id0, id1}, NcclStreamId(0), AsyncStreamKind::kCollective,
      std::vector<std::vector<GlobalDeviceId>>{{id0, id2}, {id1, id3}});
  EXPECT_FALSE(key0 == key_other);
}

TEST(NcclCliqueKeyTest, BtreeIterationOrder) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);
  GlobalDeviceId id3 = GlobalDeviceId(3);

  NcclCliqueKey key0({id0, id2}, NcclStreamId(0));
  NcclCliqueKey key1({id0, id1, id2, id3}, NcclStreamId(0));

  absl::btree_map<NcclCliqueKey, int64_t, std::greater<NcclCliqueKey>> map;
  map[key0] = 0;
  map[key1] = 1;

  EXPECT_EQ(map.begin()->first, key1);
}

TEST(NcclCliqueKeyGettersTest, Devices) {
  EXPECT_THAT(
      GetBaseCliqueKey().devices(),
      ::testing::UnorderedElementsAre(GlobalDeviceId(0), GlobalDeviceId(1)));
}

TEST(NcclCliqueKeyGettersTest, Rank) {
  auto key = GetBaseCliqueKey();
  EXPECT_EQ(key.rank(GlobalDeviceId(0)), 0);
  EXPECT_EQ(key.rank(GlobalDeviceId(1)), 1);
  EXPECT_EQ(key.rank(GlobalDeviceId(2)), std::nullopt);
  EXPECT_EQ(key.rank(GlobalDeviceId(3)), std::nullopt);
}

TEST(NcclCliqueKeyGettersTest, StreamId) {
  EXPECT_EQ(GetBaseCliqueKey().stream_id(), NcclStreamId(0));
}

TEST(NcclCliqueKeyGetterTest, ToString) {
  EXPECT_EQ(GetBaseCliqueKey().ToString(),
            "devices=[0,1]; stream=0; groups=[[0,1],[2,3]]");
}

TEST(NcclCliqueIdGettersTest, Data) {
  std::array<char, 128> id;
  std::fill(id.begin(), id.end(), 0x01);
  NcclCliqueId clique_id(id.data());
  EXPECT_EQ(std::memcmp(clique_id.data().data(), id.data(), 128), 0);
}

TEST(NcclCliqueIdStringTest, ToString) {
  std::array<char, 128> id;
  std::fill(id.begin(), id.end(), 0x01);
  NcclCliqueId clique_id(id.data());
  for (int i = 0; i < 128; ++i) {
    EXPECT_THAT(clique_id.ToString().substr(i, 1), "\x1");
  }
}

}  // namespace xla::gpu
