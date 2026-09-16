/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/service/device_assignment.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/runtime/device_id.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

TEST(DeviceAssignmentTest, Basic) {
  DeviceAssignment da(4, 2);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 2; ++c) {
      da(r, c) = c * 4 + r;
    }
  }
  EXPECT_EQ(da.ToString(),
            "DeviceAssignment{replica_count=4, computation_count=2, "
            "Computation0{0 1 2 3} Computation1{4 5 6 7}}");

  EXPECT_EQ(da(0, 0), 0);
  EXPECT_EQ(da(0, 1), 4);
  ASSERT_OK_AND_ASSIGN(auto logical_id,
                       da.LogicalIdForDevice(GlobalDeviceId(4)));
  EXPECT_EQ(logical_id.replica_id, 0);
  EXPECT_EQ(logical_id.computation_id, 1);
  EXPECT_THAT(da.LogicalIdForDevice(GlobalDeviceId(10)),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(DeviceAssignmentTest, SerDes) {
  DeviceAssignment da(4, 2);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 2; ++c) {
      da(r, c) = c * 4 + r;
    }
  }
  DeviceAssignmentProto proto;
  da.Serialize(&proto);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceAssignment> da2,
                       DeviceAssignment::Deserialize(proto));
  EXPECT_EQ(da, *da2);
}

TEST(DeviceAssignmentTest, SerDesError) {
  DeviceAssignment da(4, 2);
  DeviceAssignmentProto proto;
  da.Serialize(&proto);
  proto.set_replica_count(-1);
  EXPECT_THAT(DeviceAssignment::Deserialize(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(DeviceAssignmentTest, DuplicateDevices) {
  DeviceAssignment da(4, 2);
  da.Fill(0);
  EXPECT_EQ(da(0, 0), 0);
  EXPECT_EQ(da(0, 1), 0);
  EXPECT_THAT(da.LogicalIdForDevice(GlobalDeviceId(0)),
              StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(da.LogicalIdForDevice(GlobalDeviceId(1)),
              StatusIs(absl::StatusCode::kInternal));
}

TEST(DeviceAssignmentTest, IsIota) {
  DeviceAssignment da_empty;
  EXPECT_TRUE(da_empty.IsIota());

  DeviceAssignment da_iota(2, 2);
  da_iota(0, 0) = 0;
  da_iota(0, 1) = 1;
  da_iota(1, 0) = 2;
  da_iota(1, 1) = 3;
  EXPECT_TRUE(da_iota.IsIota());

  DeviceAssignment da_offset_iota(2, 2);
  da_offset_iota(0, 0) = 4;
  da_offset_iota(0, 1) = 5;
  da_offset_iota(1, 0) = 6;
  da_offset_iota(1, 1) = 7;
  EXPECT_TRUE(da_offset_iota.IsIota());

  DeviceAssignment da_non_iota(2, 2);
  da_non_iota(0, 0) = 1;
  da_non_iota(0, 1) = 0;
  da_non_iota(1, 0) = 2;
  da_non_iota(1, 1) = 3;
  EXPECT_FALSE(da_non_iota.IsIota());
}

TEST(DeviceAssignmentTest, IsAll) {
  DeviceAssignment da_zeros(2, 2);
  da_zeros.Fill(0);
  EXPECT_TRUE(da_zeros.IsAll(0));
  EXPECT_FALSE(da_zeros.IsAll(1));

  DeviceAssignment da_mixed(2, 2);
  da_mixed(0, 0) = 0;
  da_mixed(0, 1) = 0;
  da_mixed(1, 0) = 0;
  da_mixed(1, 1) = 1;
  EXPECT_FALSE(da_mixed.IsAll(0));
}

}  // namespace
}  // namespace xla
