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

#include "xla/service/gpu_topology.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;

TEST(GpuTopologyTest, GetGpuTopologyForPlatformTeslaA100) {
  auto topology_or = GetGpuTopologyForPlatform("tesla_a100", 1, 1, 1);
  ASSERT_OK(topology_or);
  const auto& topology = *topology_or;
  EXPECT_EQ(topology.platform_version(), "tesla_a100");
  EXPECT_EQ(topology.num_partitions(), 1);
  EXPECT_EQ(topology.num_hosts_per_partition(), 1);
  EXPECT_EQ(topology.num_devices_per_host(), 1);
  EXPECT_TRUE(topology.has_gpu_target_config());
}

TEST(GpuTopologyTest, GetGpuTopologyForPlatformNvidiaH100) {
  auto topology_or = GetGpuTopologyForPlatform("nvidia_h100", 2, 1, 4);
  ASSERT_OK(topology_or);
  const auto& topology = *topology_or;
  EXPECT_EQ(topology.platform_version(), "nvidia_h100");
  EXPECT_EQ(topology.num_partitions(), 2);
  EXPECT_EQ(topology.num_hosts_per_partition(), 1);
  EXPECT_EQ(topology.num_devices_per_host(), 4);
  EXPECT_TRUE(topology.has_gpu_target_config());
}

TEST(GpuTopologyTest, GetGpuTopologyForPlatformUmbrielB200) {
  auto topology_or = GetGpuTopologyForPlatform("umbriel_b200", 1, 2, 8);
  ASSERT_OK(topology_or);
  const auto& topology = *topology_or;
  EXPECT_EQ(topology.platform_version(), "umbriel_b200");
  EXPECT_EQ(topology.num_partitions(), 1);
  EXPECT_EQ(topology.num_hosts_per_partition(), 2);
  EXPECT_EQ(topology.num_devices_per_host(), 8);
  EXPECT_TRUE(topology.has_gpu_target_config());
}

TEST(GpuTopologyTest, GetGpuTopologyForPlatformOberonB200) {
  auto topology_or = GetGpuTopologyForPlatform("oberon_b200", 1, 2, 4);
  ASSERT_OK(topology_or);
  const auto& topology = *topology_or;
  EXPECT_EQ(topology.platform_version(), "oberon_b200");
  EXPECT_EQ(topology.num_partitions(), 1);
  EXPECT_EQ(topology.num_hosts_per_partition(), 2);
  EXPECT_EQ(topology.num_devices_per_host(), 4);
  EXPECT_TRUE(topology.has_gpu_target_config());
}

TEST(GpuTopologyTest, GetGpuTopologyForPlatformInvalid) {
  EXPECT_THAT(GetGpuTopologyForPlatform("invalid_gpu", 1, 1, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
