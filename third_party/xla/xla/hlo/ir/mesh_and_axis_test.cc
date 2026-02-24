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
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/strings/string_view.h"
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
  std::vector<absl::string_view> axes_abc = {"a", "b", "c"};
  std::vector<absl::string_view> axes_abcd = {"a", "b", "c", "d"};
  std::vector<absl::string_view> axes_efgh = {"e", "f", "g", "h"};
  EXPECT_EQ(Mesh({1, 2, 3}, axes_abc), Mesh({1, 2, 3}, axes_abc));
  EXPECT_NE(Mesh({1, 2, 3, 4}, axes_abcd), Mesh({1, 2, 3, 4}, axes_efgh));
  EXPECT_NE(Mesh({1, 2, 3}, axes_abc), Mesh({1, 2, 3, 4}, axes_abcd));
}

TEST(MeshAndAxisTest, DeviceAssignmentEquality) {
  std::vector<absl::string_view> axes_abcd = {"a", "b", "c", "d"};
  std::vector<absl::string_view> axes_efgh = {"e", "f", "g", "h"};
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

  std::vector<absl::string_view> axes_names = {"axis1", "axis2", "axis3"};
  EXPECT_THAT(
      Mesh(TileAssignment(/*dims=*/{4, 4, 1}, /*reshape_dims=*/{4, 2, 2},
                          /*transpose_perm=*/{1, 0, 2}),
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
  std::vector<absl::string_view> axes_xy = {"x", "y"};
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
  std::vector<absl::string_view> axes_xy = {"x", "y"};
  EXPECT_EQ(Mesh(array, axes_xy), Mesh::FromProto(expected));
}

TEST(MeshAndAxisTest, MeshRoundtripProto) {
  // Iota tiling.
  std::vector<absl::string_view> axes_xy = {"data", "model"};
  Mesh mesh_iota({5, 3}, axes_xy);
  EXPECT_THAT(mesh_iota, Mesh::FromProto(mesh_iota.ToProto()));

  // Non-iota tiling.
  Array2D<int64_t> array(
      {{14, 7, 6}, {12, 0, 8}, {11, 10, 5}, {1, 9, 3}, {2, 13, 4}});
  Mesh mesh_non_iota(array, axes_xy);
  EXPECT_THAT(mesh_non_iota, Mesh::FromProto(mesh_non_iota.ToProto()));
}

TEST(MeshAndAxisTest, ValidatesAxisRef) {
  EXPECT_DEATH(
      { AxisRef axis_ref_invalid_pre_size(3, {0, 2}); },
      "sub-axis pre-size must be ");
  EXPECT_DEATH(
      { AxisRef axis_ref_invalid_subaxis_size(0, {1, 1}); },
      "sub-axis size must be");
}

TEST(MeshAndAxisTest, ValidatesMeshEmptyMesh) { Mesh(); }

TEST(MeshAndAxisTest, ValidatesMeshMaximalMesh) { Mesh(5); }

TEST(MeshAndAxisTest, ValidatesMesh) {
  EXPECT_DEATH(
      { Mesh mesh_dims_axes_mismatch({2, 3, 4}, {"x", "y"}); },
      "Number of axes names must match number of dimensions in the device "
      "assignment. Number of axes names: 2, Number of dimensions: 3");

  Array2D<int64_t> negative_device_ids({{0, 1, 2}, {3, -4, 5}});
  EXPECT_DEATH(
      { Mesh mesh_invalid_non_iota(negative_device_ids, {"x", "y"}); },
      "Mesh device ids must be non-negative. Device id: -4");

  Array2D<int64_t> invalid_non_iota_device_ids({{10, 11, 12}, {13, 14, 15}});
  EXPECT_DEATH(
      { Mesh mesh_invalid_non_iota(invalid_non_iota_device_ids, {"x", "y"}); },
      "Device ids must be a permutation of");

  EXPECT_DEATH(
      {
        Mesh mesh_with_duplicate_axis_names({1, 2, 3, 4}, {"x", "y", "z", "x"});
      },
      "Mesh has duplicate axis names. Duplicate axis name: x");

  EXPECT_DEATH(
      { Mesh mesh_with_integer_axis_name({1, 2}, {"x", "1"}); },
      "Mesh axis name cannot be an integer to avoid confusion with axis "
      "indices: 1");
}

TEST(MeshAndAxisTest, FromProtoValidation) {
  {
    MeshProto proto;
    auto* axis = proto.add_axes();
    axis->set_name("x");
    axis->set_size(1);

    // 1 axis of size 1, but 2 device IDs.
    proto.add_device_ids(0);
    proto.add_device_ids(1);

    EXPECT_DEATH(
        Mesh::FromProto(proto),
        "Number of device ids must match the product of mesh axis sizes");
  }

  {
    MeshProto proto;
    auto* axis = proto.add_axes();
    axis->set_name("x");
    axis->set_size(0);

    proto.add_device_ids(0);

    EXPECT_DEATH(Mesh::FromProto(proto), "Mesh axis size must be positive");
  }

  {
    MeshProto proto;
    proto.add_device_ids(0);
    proto.add_device_ids(1);
    EXPECT_DEATH(Mesh::FromProto(proto),
                 "Maximal mesh must have exactly 1 device id");
  }
}

TEST(MeshAndAxisTest, MeshToString) {
  Mesh empty_mesh;
  EXPECT_EQ(empty_mesh.ToString(), "mesh[]");

  Mesh mesh_uvw({10, 12, 15}, {"u", "v", "w"});
  EXPECT_EQ(mesh_uvw.ToString(), "mesh[u=10,v=12,w=15]");

  Mesh mesh_abcd(
      TileAssignment(/*dims=*/{2, 4, 4, 2}, /*reshape_dims=*/{1, 4, 1, 16},
                     /*transpose_perm=*/{2, 3, 0, 1}),
      {"a", "b", "c", "d"});
  EXPECT_EQ(mesh_abcd.ToString(),
            "mesh[a=2,b=4,c=4,d=2], device_ids=([4,16]T(1,0))");

  Array<int64_t> array({{8, 3, 7, 5, 4, 2, 6, 0, 1, 9}});
  array.Reshape({10});
  Mesh mesh_ooo(array, {"ooo"});
  EXPECT_EQ(mesh_ooo.ToString(),
            "mesh[ooo=10], device_ids=(8,3,7,5,4,2,6,0,1,9)");

  Mesh maximal_mesh(5);
  EXPECT_EQ(maximal_mesh.ToString(), "maximal_mesh[device_id=5]");
}

TEST(MeshAndAxisTest, AxisRefToString) {
  EXPECT_EQ(AxisRef(1).ToString(), "1");
  EXPECT_EQ(AxisRef(2, {3, 4}).ToString(), "2:(3)4");

  Mesh mesh({10, 12, 15}, {"u", "v", "w"});
  EXPECT_EQ(AxisRef(0).ToString(&mesh), "u");
  EXPECT_EQ(AxisRef(1).ToString(&mesh), "v");
  EXPECT_EQ(AxisRef(2).ToString(&mesh), "w");
  EXPECT_EQ(AxisRef(0, {1, 2}).ToString(&mesh), "u:(1)2");
  EXPECT_EQ(AxisRef(1, {3, 4}).ToString(&mesh), "v:(3)4");
}

TEST(MeshAndAxisTest, ValidateAxisForMesh) {
  Mesh mesh({2 * 7, 3 * 11, 5 * 13}, {"fdr", "jfk", "lbj"});

  EXPECT_DEATH(
      { CHECK_OK(AxisRef(3, {1, 2}).Validate(mesh)); },
      "Axis index must be less than number of axes.*"
      "Axis index: 3, Number of axes: 3");

  EXPECT_DEATH(
      { CHECK_OK(AxisRef(0, {5, 19}).Validate(mesh)); },
      "Sub-axis next_pre_size must divide the full axis size.*"
      "Next pre-size: 95, Axis size: 14");
  EXPECT_DEATH(
      { CHECK_OK(AxisRef(0, {2, 5}).Validate(mesh)); },
      "Sub-axis next_pre_size must divide the full axis size.*"
      "Next pre-size: 10, Axis size: 14");

  EXPECT_DEATH(
      { CHECK_OK(AxisRef(1, {1, 3 * 11}).Validate(mesh)); },
      "Sub-axis size must be strictly less than the full axis size.*"
      "Sub-axis size: 33, Axis size: 33");

  AxisRefProto invalid_pre_size_proto;
  invalid_pre_size_proto.set_mesh_axis_index(0);
  invalid_pre_size_proto.mutable_sub_axis_info()->set_pre_size(0);
  invalid_pre_size_proto.mutable_sub_axis_info()->set_size(2);
  EXPECT_DEATH(
      { CHECK_OK(AxisRef::FromProto(invalid_pre_size_proto).Validate(mesh)); },
      "sub-axis pre-size must be >= 1");
}

TEST(MeshAndAxisTest, AxisRefCanCoexistWithoutOverlap) {
  auto coexistWithoutOverlap = [](AxisRef a, AxisRef b, bool expected) {
    EXPECT_EQ(a.CanCoexistWithoutOverlap(b), expected);
    EXPECT_EQ(b.CanCoexistWithoutOverlap(a), expected);
  };

  coexistWithoutOverlap(AxisRef(0), AxisRef(1), true);
  coexistWithoutOverlap(AxisRef(0), AxisRef(1, {1, 2}), true);
  coexistWithoutOverlap(AxisRef(0), AxisRef(1, {2, 2}), true);
  coexistWithoutOverlap(AxisRef(0, {1, 2}), AxisRef(0, {2, 4}), true);
  coexistWithoutOverlap(AxisRef(0, {1, 2}), AxisRef(0, {6, 2}), true);
  coexistWithoutOverlap(AxisRef(0, {1, 4}), AxisRef(0, {4, 2}), true);
  coexistWithoutOverlap(AxisRef(0, {1, 4}), AxisRef(0, {8, 2}), true);
  coexistWithoutOverlap(AxisRef(0, {4, 2}), AxisRef(0, {1, 2}), true);

  coexistWithoutOverlap(AxisRef(0), AxisRef(0), false);
  coexistWithoutOverlap(AxisRef(0), AxisRef(0, {2, 2}), false);
  coexistWithoutOverlap(AxisRef(0), AxisRef(0, {2, 4}), false);
  coexistWithoutOverlap(AxisRef(0, {2, 2}), AxisRef(0, {2, 2}), false);
  coexistWithoutOverlap(AxisRef(0, {1, 2}), AxisRef(0, {1, 4}), false);
  coexistWithoutOverlap(AxisRef(2, {1, 2}), AxisRef(2, {1, 4}), false);
  coexistWithoutOverlap(AxisRef(0, {1, 4}), AxisRef(0, {2, 2}), false);
  coexistWithoutOverlap(AxisRef(0, {1, 4}), AxisRef(0, {2, 4}), false);
  coexistWithoutOverlap(AxisRef(0, {1, 2}), AxisRef(0, {1, 3}), false);
  coexistWithoutOverlap(AxisRef(0, {1, 2}), AxisRef(0, {3, 2}), false);
  coexistWithoutOverlap(AxisRef(0, {1, 3}), AxisRef(0, {2, 3}), false);
  coexistWithoutOverlap(AxisRef(0, {2, 8}), AxisRef(0, {4, 2}), false);
}

TEST(MeshAndAxisTest, EmptyMesh) {
  Mesh empty_mesh;
  EXPECT_EQ(empty_mesh, Mesh());
  EXPECT_NE(empty_mesh, Mesh(5));
  EXPECT_NE(empty_mesh, Mesh({1}, {"a"}));
  EXPECT_FALSE(empty_mesh.IsMaximal());
  EXPECT_THAT(empty_mesh.ToProto(), EqualsProto(MeshProto()));
  EXPECT_EQ(empty_mesh, Mesh::FromProto(MeshProto()));
  EXPECT_EQ(empty_mesh, Mesh::FromProto(empty_mesh.ToProto()));
}

TEST(MeshAndAxisTest, MaximalMesh) {
  Mesh maximal_mesh(5);
  EXPECT_TRUE(maximal_mesh.IsMaximal());
  Mesh non_maximal_mesh({2, 3}, {"a", "b"});
  EXPECT_FALSE(non_maximal_mesh.IsMaximal());
  Mesh mesh_single_axis({1}, {"a"});
  EXPECT_FALSE(mesh_single_axis.IsMaximal());

  EXPECT_EQ(maximal_mesh, Mesh(5));
  EXPECT_NE(maximal_mesh, Mesh(6));

  MeshProto expected_proto;
  expected_proto.add_device_ids(5);
  EXPECT_THAT(maximal_mesh.ToProto(), EqualsProto(expected_proto));

  MeshProto from_proto;
  from_proto.add_device_ids(7);
  EXPECT_EQ(Mesh(7), Mesh::FromProto(from_proto));

  EXPECT_EQ(maximal_mesh, Mesh::FromProto(maximal_mesh.ToProto()));
}

TEST(MeshAndAxisTest, AxisRefSize) {
  Mesh mesh({2 * 7, 3 * 11, 5 * 13}, {"a", "b", "c"});
  EXPECT_EQ(AxisRef(0).size(mesh), 14);
  EXPECT_EQ(AxisRef(1).size(mesh), 33);
  EXPECT_EQ(AxisRef(2).size(mesh), 65);
  EXPECT_EQ(AxisRef(0, {1, 2}).size(mesh), 2);
  EXPECT_EQ(AxisRef(0, {2, 7}).size(mesh), 7);
  EXPECT_EQ(AxisRef(1, {1, 3}).size(mesh), 3);
  EXPECT_EQ(AxisRef(1, {3, 11}).size(mesh), 11);
  EXPECT_EQ(AxisRef(2, {1, 5}).size(mesh), 5);
  EXPECT_EQ(AxisRef(2, {5, 13}).size(mesh), 13);
}

TEST(MeshAndAxisTest, AxisRefCanMerge) {
  auto checkCanMerge = [](AxisRef a, AxisRef b) {
    EXPECT_TRUE(a.CanMerge(b));
    EXPECT_FALSE(b.CanMerge(a));
  };

  checkCanMerge(AxisRef(0, {1, 2}), AxisRef(0, {2, 4}));
  checkCanMerge(AxisRef(0, {2, 4}), AxisRef(0, {8, 2}));

  EXPECT_FALSE(AxisRef(0, {1, 2}).CanMerge(AxisRef(0, {1, 2})));
  EXPECT_FALSE(AxisRef(0, {1, 2}).CanMerge(AxisRef(0, {4, 2})));
  EXPECT_FALSE(AxisRef(0).CanMerge(AxisRef(0, {1, 2})));
  EXPECT_FALSE(AxisRef(0, {1, 2}).CanMerge(AxisRef(0)));
  EXPECT_FALSE(AxisRef(0).CanMerge(AxisRef(0)));
  EXPECT_FALSE(AxisRef(0).CanMerge(AxisRef(1)));
  EXPECT_FALSE(AxisRef(0, {1, 2}).CanMerge(AxisRef(1, {2, 4})));
}

TEST(MeshAndAxisTest, AxisRefMerge) {
  Mesh mesh({16}, {"a"});

  AxisRef axis_ref1(0, {1, 2});
  EXPECT_TRUE(axis_ref1.Merge(AxisRef(0, {2, 4}), mesh));
  EXPECT_EQ(axis_ref1, AxisRef(0, {1, 8}));

  AxisRef axis_ref2(0, {2, 2});
  EXPECT_TRUE(axis_ref2.Merge(AxisRef(0, {4, 4}), mesh));
  EXPECT_EQ(axis_ref2, AxisRef(0, {2, 8}));

  AxisRef axis_ref3(0, {1, 8});
  EXPECT_TRUE(axis_ref3.Merge(AxisRef(0, {8, 2}), mesh));
  EXPECT_EQ(axis_ref3, AxisRef(0));

  AxisRef axis_ref4(0, {2, 4});
  EXPECT_FALSE(axis_ref4.Merge(AxisRef(0, {1, 2}), mesh));
  EXPECT_EQ(axis_ref4, AxisRef(0, {2, 4}));
}

}  // namespace xla
