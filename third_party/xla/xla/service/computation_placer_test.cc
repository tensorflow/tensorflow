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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/service/device_assignment.h"

namespace xla {
namespace {

TEST(ComputationPlacerTest, Basic) {
  ComputationPlacer cp;
  ASSERT_OK_AND_ASSIGN(DeviceAssignment da, cp.AssignDevices(4, 2));
  EXPECT_EQ(da.ToString(),
            "DeviceAssignment{replica_count=4, computation_count=2, "
            "Computation0{0 1 2 3} Computation1{4 5 6 7}}");

  EXPECT_EQ(da(0, 0), 0);
  EXPECT_EQ(da(0, 1), 4);
}

}  // namespace
}  // namespace xla
