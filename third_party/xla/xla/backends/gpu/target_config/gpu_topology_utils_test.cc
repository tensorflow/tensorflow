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

#include "xla/backends/gpu/target_config/gpu_topology_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(IsCompatibleWithTargetTopologyTest, SamePlatformsAreCompatible) {
  GpuTopology single_topology("platform_version", 1, 1, 1);
  EXPECT_THAT(IsCompatibleWithTargetTopology(single_topology, single_topology),
              IsOkAndHolds(true));
}

TEST(IsCompatibleWithTargetTopologyTest, B200andGB200AreCompatible) {
  ASSERT_OK_AND_ASSIGN(GpuTopology b200_topology,
                       GetGpuTopologyForPlatform("umbriel_b200", 1, 1, 1));
  ASSERT_OK_AND_ASSIGN(GpuTopology gb200_topology,
                       GetGpuTopologyForPlatform("oberon_b200", 1, 1, 1));
  EXPECT_THAT(IsCompatibleWithTargetTopology(b200_topology, gb200_topology),
              IsOkAndHolds(true));
  // Compatibility is symmetric for B200 and GB200.
  EXPECT_THAT(IsCompatibleWithTargetTopology(gb200_topology, b200_topology),
              IsOkAndHolds(true));
}

TEST(IsCompatibleWithTargetTopologyTest, B200andH100AreIncompatible) {
  ASSERT_OK_AND_ASSIGN(GpuTopology b200_topology,
                       GetGpuTopologyForPlatform("umbriel_b200", 1, 1, 1));
  ASSERT_OK_AND_ASSIGN(GpuTopology h100_topology,
                       GetGpuTopologyForPlatform("nvidia_h100", 1, 1, 1));
  EXPECT_THAT(IsCompatibleWithTargetTopology(b200_topology, h100_topology),
              IsOkAndHolds(false));
  EXPECT_THAT(IsCompatibleWithTargetTopology(h100_topology, b200_topology),
              IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
