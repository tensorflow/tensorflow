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

#include "xla/service/computation_placer.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/runtime/device_id.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

TEST(ComputationPlacerTest, Basic) {
  ComputationPlacer cp;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment da, cp.AssignDevices(4, 2));
  EXPECT_EQ(da.ToString(),
            "DeviceAssignment{replica_count=4, computation_count=2, "
            "Computation0{0 1 2 3} Computation1{4 5 6 7}}");

  EXPECT_EQ(da(0, 0), 0);
  EXPECT_EQ(da(0, 1), 4);
  TF_ASSERT_OK_AND_ASSIGN(auto logical_id,
                          da.LogicalIdForDevice(GlobalDeviceId(4)));
  EXPECT_EQ(logical_id.replica_id, 0);
  EXPECT_EQ(logical_id.computation_id, 1);
  EXPECT_FALSE(da.LogicalIdForDevice(GlobalDeviceId(10)).ok());
}

TEST(ComputationPlacerTest, SerDes) {
  ComputationPlacer cp;
  TF_ASSERT_OK_AND_ASSIGN(DeviceAssignment da, cp.AssignDevices(4, 2));
  DeviceAssignmentProto proto;
  da.Serialize(&proto);
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<DeviceAssignment> da2,
                          DeviceAssignment::Deserialize(proto));
  EXPECT_EQ(da, *da2);
}

TEST(ComputationPlacerTest, DuplicateDevices) {
  DeviceAssignment da(4, 2);
  da.Fill(0);
  EXPECT_EQ(da(0, 0), 0);
  EXPECT_EQ(da(0, 1), 0);
  EXPECT_FALSE(da.LogicalIdForDevice(GlobalDeviceId(0)).ok());
  EXPECT_FALSE(da.LogicalIdForDevice(GlobalDeviceId(1)).ok());
}

}  // namespace
}  // namespace xla
