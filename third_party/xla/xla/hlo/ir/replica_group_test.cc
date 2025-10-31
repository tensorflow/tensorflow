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

TEST(MeshAxesReplicaGroupListTest, ReplicaGroupsCountAndSize) {
  Mesh all_axes(TileAssignment(IotaTileAssignment::Create(
                    /*dims=*/{4, 4})),
                /*axes_names=*/{"x", "y"});
  MeshAxesReplicaGroupList replica_group_across_all_axes(
      all_axes,
      /*axes=*/{AxisRef(0), AxisRef(1)});
  EXPECT_EQ(replica_group_across_all_axes.num_replica_groups(), 1);
  EXPECT_EQ(replica_group_across_all_axes.num_devices_per_group(), 16);

  Mesh one_axes(TileAssignment(IotaTileAssignment::Create(
                    /*dims=*/{3, 5})),
                /*axes_names=*/{"a", "b"});
  MeshAxesReplicaGroupList replica_group_across_a(one_axes,
                                                  /*axes=*/{AxisRef(0)});
  MeshAxesReplicaGroupList replica_group_across_b(one_axes,
                                                  /*axes=*/{AxisRef(1)});
  EXPECT_EQ(replica_group_across_a.num_replica_groups(), 5);
  EXPECT_EQ(replica_group_across_a.num_devices_per_group(), 3);
  EXPECT_EQ(replica_group_across_b.num_replica_groups(), 3);
  EXPECT_EQ(replica_group_across_b.num_devices_per_group(), 5);

  Mesh no_axes(TileAssignment(IotaTileAssignment::Create(
                   /*dims=*/{2, 3, 5})),
               /*axes_names=*/{"p1", "p2", "p3"});
  MeshAxesReplicaGroupList replica_group_across_no_axes(no_axes, /*axes=*/{});
  EXPECT_EQ(replica_group_across_no_axes.num_replica_groups(), 2 * 3 * 5);
  EXPECT_EQ(replica_group_across_no_axes.num_devices_per_group(), 1);
}

TEST(MeshAxesReplicaGroupListTest, ReplicaGroupsCountAndSizeForSubaxes) {
  Mesh mesh_one_subaxis(TileAssignment(IotaTileAssignment::Create(
                            /*dims=*/{2, 6, 10})),
                        /*axes_names=*/{"axis1", "axis2", "axis3"});
  MeshAxesReplicaGroupList replica_group_across_axis1_subaxis(
      mesh_one_subaxis,
      /*axes=*/{AxisRef(0, {1, 2})});
  MeshAxesReplicaGroupList replica_group_across_axis2_subaxis(
      mesh_one_subaxis,
      /*axes=*/{AxisRef(1, {2, 3})});
  EXPECT_EQ(replica_group_across_axis1_subaxis.num_replica_groups(), 60);
  EXPECT_EQ(replica_group_across_axis1_subaxis.num_devices_per_group(), 2);
  EXPECT_EQ(replica_group_across_axis2_subaxis.num_replica_groups(), 40);
  EXPECT_EQ(replica_group_across_axis2_subaxis.num_devices_per_group(), 3);

  Mesh mesh_multiple_subaxis(TileAssignment(IotaTileAssignment::Create(
                                 /*dims=*/{2 * 3, 5 * 7, 11 * 13})),
                             /*axes_names=*/{"alpha", "beta", "gamma"});
  MeshAxesReplicaGroupList replica_group_across_multiple_subaxis1(
      mesh_multiple_subaxis,
      /*axes=*/{AxisRef(0, {1, 2}), AxisRef(1, {1, 5}), AxisRef(2, {1, 11})});
  MeshAxesReplicaGroupList replica_group_across_multiple_subaxis2(
      mesh_multiple_subaxis,
      /*axes=*/{AxisRef(0, {2, 3}), AxisRef(1, {5, 7}), AxisRef(2, {11, 13})});
  EXPECT_EQ(replica_group_across_multiple_subaxis1.num_replica_groups(),
            3 * 7 * 13);
  EXPECT_EQ(replica_group_across_multiple_subaxis1.num_devices_per_group(),
            2 * 5 * 11);
  EXPECT_EQ(replica_group_across_multiple_subaxis2.num_replica_groups(),
            2 * 5 * 11);
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
            "@mesh<a=2,b=4,c=4,d=2>([4,16]T(1,0)) {}");
  MeshAxesReplicaGroupList rg_abcd_across_multiple_axes_and_subaxes(
      mesh_abcd, /*axes=*/{AxisRef(0), AxisRef(1, {1, 2}), AxisRef(3)});
  EXPECT_EQ(rg_abcd_across_multiple_axes_and_subaxes.ToString(),
            "@mesh<a=2,b=4,c=4,d=2>([4,16]T(1,0)) {a,b:(1)2,d}");

  // Subaxes and random device assignment.
  Array<int64_t> array({{8, 3, 7, 5, 4, 2, 6, 0, 1, 9}});
  array.Reshape({10});
  TileAssignment tile_assignment(std::make_shared<Array<int64_t>>(array));
  Mesh mesh_ooo(tile_assignment, /*axes_names=*/{"ooo"});
  MeshAxesReplicaGroupList rg_ooo_across_none(mesh_ooo, /*axes=*/{});
  EXPECT_EQ(rg_ooo_across_none.ToString(),
            "@mesh<ooo=10>(8,3,7,5,4,2,6,0,1,9) {}");
  MeshAxesReplicaGroupList rg_ooo_across_ooo_5_2(mesh_ooo,
                                                 /*axes=*/{AxisRef(0, {5, 2})});
  EXPECT_EQ(rg_ooo_across_ooo_5_2.ToString(),
            "@mesh<ooo=10>(8,3,7,5,4,2,6,0,1,9) {ooo:(5)2}");
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
