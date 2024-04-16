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

#include "xla/service/gpu/nccl_clique_key.h"

#include <cstdint>
#include <functional>

#include "absl/container/btree_map.h"
#include "xla/service/global_device_id.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

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

  EXPECT_LT(key0, key1);
  EXPECT_GT(key1, key0);
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

}  // namespace xla::gpu
