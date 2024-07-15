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

#include "xla/hlo/ir/collective_device_list.h"

#include <gtest/gtest.h>

namespace xla {

TEST(CollectiveDeviceListTest, DefaultListToString) {
  CollectiveDeviceList list({{1, 2}, {3, 4}});
  ASSERT_EQ(list.ToString(), "{{1,2},{3,4}}");
}

TEST(CollectiveDeviceListTest, DefaultListToString2) {
  CollectiveDeviceList list({{1, 2, 3, 4, 5, 6, 7}});
  EXPECT_EQ(list.ToString(), "{{1,2,3,4,5,6,7}}");
}

TEST(CollectiveDeviceListTest, IotaListToString) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  EXPECT_EQ(list.ToString(), "[2,10]<=[20]");
}

TEST(CollectiveDeviceListTest, IotaListToString2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  EXPECT_EQ(list.ToString(), "[2,10]<=[4,5]T(1,0)");
}

}  // namespace xla
