/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/ir/mesh_and_axis.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/array2d.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/tsl/util/proto/proto_matchers.h"

using ::tsl::proto_testing::EqualsProto;

namespace xla {

TEST(MeshAndAxisTest, AxisRefEquality) {
  EXPECT_EQ(AxisRef(1), AxisRef(1));
  EXPECT_EQ(AxisRef(3, {1, 2}), AxisRef(3, {1, 2}));
  EXPECT_NE(AxisRef(2), AxisRef(4));
  EXPECT_NE(AxisRef(0), AxisRef(0, {1, 2}));
  EXPECT_NE(AxisRef(2, {1, 2}), AxisRef(3, {1, 2}));
  EXPECT_NE(AxisRef(2, {1, 2}), AxisRef(2, {2, 2}));
  EXPECT_NE(AxisRef(2, {1, 2}), AxisRef(2, {1, 4}));
}

TEST(MeshAndAxisTest, MeshEquality) {
  std::vector<std::string> axes_abc = {"a", "b", "c"};
  std::vector<std::string> axes_abcd = {"a", "b", "c", "d"};
  std::vector<std::string> axes_efgh = {"e", "f", "g", "h"};
  EXPECT_EQ(Mesh({1, 2, 3}, axes_abc), Mesh({1, 2, 3}, axes_abc));
  EXPECT_NE(Mesh({1, 2, 3, 4}, axes_abcd), Mesh({1, 2, 3, 4}, axes_efgh));
  EXPECT_NE(Mesh({1, 2, 3}, axes_abc), Mesh({1, 2, 3, 4}, axes_abcd));
}

TEST(MeshAndAxisTest, DeviceAssignmentEquality) {
  std::vector<std::string> axes_abcd = {"a", "b", "c", "d"};
  std::vector<std::string> axes_efgh = {"e", "f", "g", "h"};
  Mesh mesh({1, 2, 3, 4}, axes_abcd);
  Mesh mesh_diff_axis_names({1, 2, 3, 4}, axes_efgh);
  EXPECT_TRUE(mesh.DeviceAssignmentEquals(mesh_diff_axis_names));
  Mesh mesh_other({2, 1, 4, 3}, axes_efgh);
  EXPECT_FALSE(mesh.DeviceAssignmentEquals(mesh_other));
}

TEST(MeshAndAxisTest, AxesToProto) {
  AxisRefProto expected;
  expected.set_mesh_axis_index(123);
  EXPECT_THAT(AxisRef(123).ToProto(), EqualsProto(expected));
}

TEST(MeshAndAxisTest, AxesToProtoWithSubAxis) {
  AxisRefProto expected;
  expected.set_mesh_axis_index(2);
  expected.mutable_sub_axis_info()->set_pre_size(2);
  expected.mutable_sub_axis_info()->set_size(8);
  EXPECT_THAT(AxisRef(2, {2, 8}).ToProto(), EqualsProto(expected));
}

TEST(MeshAndAxisTest, AxesFromProto) {
  AxisRefProto expected;
  expected.set_mesh_axis_index(1);
  EXPECT_THAT(AxisRef(1), AxisRef::FromProto(expected));
}

TEST(MeshAndAxisTest, AxesFromProtoWithSubAxis) {
  AxisRefProto expected;
  expected.set_mesh_axis_index(10);
  expected.mutable_sub_axis_info()->set_pre_size(4);
  expected.mutable_sub_axis_info()->set_size(32);
  EXPECT_THAT(AxisRef(10, {4, 32}), AxisRef::FromProto(expected));
}

TEST(MeshAndAxisTest, MeshToAndFromProtoIotaTiling) {
  MeshProto proto;
  proto.add_axes()->set_name("a");
  proto.add_axes()->set_name("b");
  proto.add_axes()->set_name("c");
  proto.mutable_axes(0)->set_size(2);
  proto.mutable_axes(1)->set_size(3);
  proto.mutable_axes(2)->set_size(6);

  Mesh mesh({2, 3, 6}, {"a", "b", "c"});

  EXPECT_THAT(mesh.ToProto(), EqualsProto(proto));
  EXPECT_EQ(mesh, Mesh::FromProto(proto));
}

TEST(MeshAndAxisTest, MeshToProtoIotaTilingWithReshapeDims) {
  MeshProto expected;
  expected.add_axes()->set_name("axis1");
  expected.add_axes()->set_name("axis2");
  expected.add_axes()->set_name("axis3");
  expected.mutable_axes(0)->set_size(4);
  expected.mutable_axes(1)->set_size(4);
  expected.mutable_axes(2)->set_size(1);
  // When dims=[4,4,1] reshape_dims=[4,2,2], transpose_perm=[1,0,2] (swap dim 0
  // and dim 1) corresponds to [4,4,1]<=[4,2,2]T(1,0,2) which in full array V1
  // format is [0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15].
  std::vector<int> expected_device_ids = {0, 1, 4, 5, 8,  9,  12, 13,
                                          2, 3, 6, 7, 10, 11, 14, 15};
  for (int i = 0; i < expected_device_ids.size(); ++i) {
    expected.add_device_ids(expected_device_ids[i]);
  }

  std::vector<std::string> axes_names = {"axis1", "axis2", "axis3"};
  EXPECT_THAT(
      Mesh(TileAssignment(IotaTileAssignment::Create(
               /*dims=*/{4, 4, 1},
               /*reshape_dims=*/{4, 2, 2}, /*transpose_perm=*/{1, 0, 2})),
           axes_names)
          .ToProto(),
      EqualsProto(expected));
}

TEST(MeshAndAxisTest, MeshToProtoNonIotaTiling) {
  MeshProto expected;
  expected.add_axes()->set_name("x");
  expected.add_axes()->set_name("y");
  expected.mutable_axes(0)->set_size(4);
  expected.mutable_axes(1)->set_size(2);
  std::vector<int> random_expected_device_ids = {6, 3, 0, 1, 5, 2, 7, 4};
  for (int i = 0; i < random_expected_device_ids.size(); ++i) {
    expected.add_device_ids(random_expected_device_ids[i]);
  }

  Array2D<int64_t> array({{6, 3}, {0, 1}, {5, 2}, {7, 4}});
  std::vector<std::string> axes_xy = {"x", "y"};
  EXPECT_THAT(Mesh(array, axes_xy).ToProto(), EqualsProto(expected));
}

TEST(MeshAndAxisTest, MeshFromProtoNonIotaTiling) {
  MeshProto expected;
  expected.add_axes()->set_name("x");
  expected.add_axes()->set_name("y");
  expected.mutable_axes(0)->set_size(4);
  expected.mutable_axes(1)->set_size(2);
  std::vector<int> random_expected_device_ids = {0, 1, 6, 3, 7, 4, 5, 2};
  for (int i = 0; i < random_expected_device_ids.size(); ++i) {
    expected.add_device_ids(random_expected_device_ids[i]);
  }

  Array2D<int64_t> array({{0, 1}, {6, 3}, {7, 4}, {5, 2}});
  std::vector<std::string> axes_xy = {"x", "y"};
  EXPECT_EQ(Mesh(array, axes_xy), Mesh::FromProto(expected));
}

TEST(MeshAndAxisTest, MeshRoundtripProto) {
  // Iota tiling.
  std::vector<std::string> axes_xy = {"data", "model"};
  Mesh mesh_iota({5, 3}, axes_xy);
  EXPECT_THAT(mesh_iota, Mesh::FromProto(mesh_iota.ToProto()));

  // Non-iota tiling.
  Array2D<int64_t> array(
      {{14, 7, 6}, {12, 0, 8}, {11, 10, 5}, {11, 9, 3}, {2, 13, 4}});
  Mesh mesh_non_iota(array, axes_xy);
  EXPECT_THAT(mesh_non_iota, Mesh::FromProto(mesh_non_iota.ToProto()));
}

TEST(MeshAxesReplicaGroupListTest, MeshAxesToString) {
  Mesh mesh_uvw({10, 12, 15}, {"u", "v", "w"});
  EXPECT_EQ(mesh_uvw.ToString(), "@mesh<u=10,v=12,w=15>");

  Mesh mesh_abcd(TileAssignment(IotaTileAssignment::Create(
                     /*dims=*/{2, 4, 4, 2}, /*reshape_dims=*/{1, 4, 1, 16},
                     /*transpose_perm=*/{2, 3, 0, 1})),
                 /*axes_names=*/{"a", "b", "c", "d"});
  EXPECT_EQ(mesh_abcd.ToString(), "@mesh<a=2,b=4,c=4,d=2>([4,16]T(1,0))");

  Array<int64_t> array({{8, 3, 7, 5, 4, 2, 6, 0, 1, 9}});
  array.Reshape({10});
  Mesh mesh_ooo(array, /*axes_names=*/{"ooo"});
  EXPECT_EQ(mesh_ooo.ToString(), "@mesh<ooo=10>(8,3,7,5,4,2,6,0,1,9)");
}

}  // namespace xla
