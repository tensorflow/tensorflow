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

#include <memory>
#include <optional>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Optional;
using ::testing::Property;
using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;

TEST(GpuTopologyTest, StoresHostTargetMachineOptions) {
  cpu::TargetMachineOptionsProto proto;
  proto.set_triple("some_triple");
  proto.set_cpu("some_cpu");
  proto.set_features("+some_feature,-some_other_feature");
  ASSERT_OK_AND_ASSIGN(cpu::TargetMachineOptions options,
                       cpu::TargetMachineOptions::FromProto(proto));
  GpuTopology topology("some_platform_version", 1, 1, 1, std::nullopt, options);
  ASSERT_TRUE(topology.host_target_machine_options().has_value());
  EXPECT_EQ(topology.host_target_machine_options()->triple(), "some_triple");
  EXPECT_EQ(topology.host_target_machine_options()->cpu(), "some_cpu");
  EXPECT_THAT(topology.host_target_machine_options()->enabled_features(),
              UnorderedElementsAre("some_feature"));
  EXPECT_THAT(topology.host_target_machine_options()->disabled_features(),
              UnorderedElementsAre("some_other_feature"));
}

TEST(GpuTopologyTest, FromProto) {
  GpuTopologyProto gpu_topology_proto;
  gpu_topology_proto.set_platform_version("some_platform_version");
  gpu_topology_proto.set_num_partitions(1);
  gpu_topology_proto.set_num_hosts_per_partition(2);
  gpu_topology_proto.set_num_devices_per_host(3);
  gpu_topology_proto.mutable_gpu_target_config()
      ->mutable_gpu_device_info()
      ->mutable_cuda_compute_capability()
      ->set_feature_extension(
          stream_executor::CudaComputeCapabilityProto::NONE);
  gpu_topology_proto.mutable_gpu_target_config()->mutable_dnn_version_info();
  gpu_topology_proto.mutable_gpu_target_config()->mutable_runtime_version();
  *gpu_topology_proto.mutable_host_target_machine_options() =
      xla::cpu::TargetMachineOptionsProto();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<const GpuTopology> topology,
                       GpuTopology::FromProto(gpu_topology_proto));
  EXPECT_EQ(topology->platform_version(), "some_platform_version");
  EXPECT_EQ(topology->num_partitions(), 1);
  EXPECT_EQ(topology->num_hosts_per_partition(), 2);
  EXPECT_EQ(topology->num_devices_per_host(), 3);
  EXPECT_TRUE(topology->has_gpu_target_config());
  EXPECT_TRUE(topology->host_target_machine_options().has_value());
  EXPECT_THAT(topology->ToProto(), EqualsProto(gpu_topology_proto));
}

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
  EXPECT_THAT(topology.host_target_machine_options(),
              Optional(Property(&cpu::TargetMachineOptions::triple,
                                "x86_64-grtev4-linux-gnu")));
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
  EXPECT_THAT(topology.host_target_machine_options(),
              Optional(Property(&cpu::TargetMachineOptions::triple,
                                "aarch64-linux-gnu")));
}

TEST(GpuTopologyTest, GetGpuTopologyForPlatformOberonB300) {
  auto topology_or = GetGpuTopologyForPlatform("oberon_b300", 1, 2, 4);
  ASSERT_OK(topology_or);
  const auto& topology = *topology_or;
  EXPECT_EQ(topology.platform_version(), "oberon_b300");
  EXPECT_EQ(topology.num_partitions(), 1);
  EXPECT_EQ(topology.num_hosts_per_partition(), 2);
  EXPECT_EQ(topology.num_devices_per_host(), 4);
  EXPECT_TRUE(topology.has_gpu_target_config());
  EXPECT_THAT(topology.host_target_machine_options(),
              Optional(Property(&cpu::TargetMachineOptions::triple,
                                "aarch64-linux-gnu")));
}

TEST(GpuTopologyTest, GetGpuTopologyForPlatformInvalid) {
  EXPECT_THAT(GetGpuTopologyForPlatform("invalid_gpu", 1, 1, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
