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

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
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

TEST(CollectiveDeviceListTest, DefaultListToString) {
  CollectiveDeviceList list({{1, 2}, {3, 4}});
  ASSERT_EQ(list.ToString(), "{{1,2},{3,4}}");
}

TEST(CollectiveDeviceListTest, DefaultListToString2) {
  CollectiveDeviceList list({{1, 2, 3, 4, 5, 6, 7}});
  EXPECT_EQ(list.ToString(), "{{1,2,3,4,5,6,7}}");
}

TEST(CollectiveDeviceListTest, DefaultListToProto) {
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

TEST(CollectiveDeviceListTest, IotaListToString) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  EXPECT_EQ(list.ToString(), "[2,10]<=[20]");
}

TEST(CollectiveDeviceListTest,
     IotaListToStringWithPrintingFullReplicaGroupList) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10));
  EXPECT_EQ(list.ToString(/*print_full_replica_group_list=*/true),
            "{{0,1,2,3,4,5,6,7,8,9},{10,11,12,13,14,15,16,17,18,19}}");
}

TEST(CollectiveDeviceListTest, IotaListToString2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  EXPECT_EQ(list.ToString(), "[2,10]<=[4,5]T(1,0)");
}

TEST(CollectiveDeviceListTest,
     IotaListToStringWithPrintingFullReplicaGroupList2) {
  CollectiveDeviceList list(IotaReplicaGroupList(2, 10, {4, 5}, {1, 0}));
  EXPECT_EQ(list.ToString(/*print_full_replica_group_list=*/true),
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
