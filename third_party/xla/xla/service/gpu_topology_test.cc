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
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/cpu/executable.pb.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.pb.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "tsl/platform/path.h"

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
  gpu_topology_proto.mutable_gpu_target_config()
      ->mutable_gpu_device_info()
      ->set_dnn_version("9.0.0");
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
  // Default value for num_devices_per_process is num_devices_per_host.
  EXPECT_EQ(topology->num_devices_per_process(), 3);
  // But original value is not set.
  EXPECT_EQ(topology->num_devices_per_process_opt(), std::nullopt);
  EXPECT_THAT(topology->ToProto(), EqualsProto(gpu_topology_proto));
}

TEST(GpuTopologyTest, FromProtoWithNumDevicesPerProcess) {
  GpuTopologyProto gpu_topology_proto;
  gpu_topology_proto.set_platform_version("some_platform_version");
  gpu_topology_proto.set_num_partitions(1);
  gpu_topology_proto.set_num_hosts_per_partition(2);
  gpu_topology_proto.set_num_devices_per_host(4);
  gpu_topology_proto.set_num_devices_per_process(2);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<const GpuTopology> topology,
                       GpuTopology::FromProto(gpu_topology_proto));
  EXPECT_EQ(topology->platform_version(), "some_platform_version");
  EXPECT_EQ(topology->num_partitions(), 1);
  EXPECT_EQ(topology->num_hosts_per_partition(), 2);
  EXPECT_EQ(topology->num_devices_per_host(), 4);
  EXPECT_THAT(topology->ToProto(), EqualsProto(gpu_topology_proto));
  EXPECT_EQ(topology->num_devices_per_process(), 2);
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

TEST(GpuTopologyTest, GetGpuTopologyForPlatformTeslaH200) {
  auto topology_or = GetGpuTopologyForPlatform("tesla_h200", 2, 1, 4);
  ASSERT_OK(topology_or);
  const auto& topology = *topology_or;
  EXPECT_EQ(topology.platform_version(), "tesla_h200");
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
                                "aarch64-unknown-linux-gnu")));
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
                                "aarch64-unknown-linux-gnu")));
}

TEST(GpuTopologyTest, GetGpuTopologyForPlatformInvalid) {
  EXPECT_THAT(GetGpuTopologyForPlatform("invalid_gpu", 1, 1, 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(GpuTopologyTest, ParseGpuTopologyShorthand) {
  ASSERT_OK_AND_ASSIGN(GpuTopology dims_only_3d, ParseGpuTopology("1x2x8"));
  EXPECT_EQ(dims_only_3d.platform_version(), "");
  EXPECT_EQ(dims_only_3d.num_partitions(), 1);
  EXPECT_EQ(dims_only_3d.num_hosts_per_partition(), 2);
  EXPECT_EQ(dims_only_3d.num_devices_per_host(), 8);
  EXPECT_FALSE(dims_only_3d.has_gpu_target_config());

  ASSERT_OK_AND_ASSIGN(GpuTopology dims_only_2d, ParseGpuTopology("2x8"));
  EXPECT_EQ(dims_only_2d.platform_version(), "");
  EXPECT_EQ(dims_only_2d.num_partitions(), 1);
  EXPECT_EQ(dims_only_2d.num_hosts_per_partition(), 2);
  EXPECT_EQ(dims_only_2d.num_devices_per_host(), 8);
  EXPECT_FALSE(dims_only_2d.has_gpu_target_config());

  ASSERT_OK_AND_ASSIGN(GpuTopology with_platform_3d,
                       ParseGpuTopology("nvidia_h100:1x2x8"));
  EXPECT_EQ(with_platform_3d.platform_version(), "nvidia_h100");
  EXPECT_EQ(with_platform_3d.num_partitions(), 1);
  EXPECT_EQ(with_platform_3d.num_hosts_per_partition(), 2);
  EXPECT_EQ(with_platform_3d.num_devices_per_host(), 8);
  EXPECT_TRUE(with_platform_3d.has_gpu_target_config());

  ASSERT_OK_AND_ASSIGN(GpuTopology with_platform_2d,
                       ParseGpuTopology("oberon_b300:4x4"));
  EXPECT_EQ(with_platform_2d.platform_version(), "oberon_b300");
  EXPECT_EQ(with_platform_2d.num_partitions(), 1);
  EXPECT_EQ(with_platform_2d.num_hosts_per_partition(), 4);
  EXPECT_EQ(with_platform_2d.num_devices_per_host(), 4);
  EXPECT_TRUE(with_platform_2d.has_gpu_target_config());
  EXPECT_THAT(with_platform_2d.host_target_machine_options(),
              Optional(Property(&cpu::TargetMachineOptions::triple,
                                "aarch64-unknown-linux-gnu")));
}

TEST(GpuTopologyTest, ParseGpuTopologySpecFiles) {
  auto spec_path = [](absl::string_view file_name) {
    return tsl::io::JoinPath(tsl::testing::XlaSrcRoot(),
                             "backends/gpu/target_config/specs", file_name);
  };

  ASSERT_OK_AND_ASSIGN(GpuTopology h100_1x8,
                       ParseGpuTopology(spec_path("h100_1x8.txtpb")));
  EXPECT_EQ(h100_1x8.platform_version(), "nvidia_h100");
  EXPECT_EQ(h100_1x8.num_partitions(), 1);
  EXPECT_EQ(h100_1x8.num_hosts_per_partition(), 1);
  EXPECT_EQ(h100_1x8.num_devices_per_host(), 8);
  EXPECT_TRUE(h100_1x8.has_gpu_target_config());

  ASSERT_OK_AND_ASSIGN(GpuTopology h100_2x8,
                       ParseGpuTopology(spec_path("h100_2x8.txtpb")));
  EXPECT_EQ(h100_2x8.platform_version(), "nvidia_h100");
  EXPECT_EQ(h100_2x8.num_partitions(), 1);
  EXPECT_EQ(h100_2x8.num_hosts_per_partition(), 2);
  EXPECT_EQ(h100_2x8.num_devices_per_host(), 8);
  EXPECT_TRUE(h100_2x8.has_gpu_target_config());

  ASSERT_OK_AND_ASSIGN(GpuTopology gb200_2x4,
                       ParseGpuTopology(spec_path("gb200_2x4.txtpb")));
  EXPECT_EQ(gb200_2x4.platform_version(), "oberon_b200");
  EXPECT_EQ(gb200_2x4.num_partitions(), 1);
  EXPECT_EQ(gb200_2x4.num_hosts_per_partition(), 2);
  EXPECT_EQ(gb200_2x4.num_devices_per_host(), 4);
  EXPECT_TRUE(gb200_2x4.has_gpu_target_config());
  EXPECT_THAT(gb200_2x4.host_target_machine_options(),
              Optional(Property(&cpu::TargetMachineOptions::triple,
                                "aarch64-unknown-linux-gnu")));

  ASSERT_OK_AND_ASSIGN(GpuTopology gb300_4x4,
                       ParseGpuTopology(spec_path("gb300_4x4.txtpb")));
  EXPECT_EQ(gb300_4x4.platform_version(), "oberon_b300");
  EXPECT_EQ(gb300_4x4.num_partitions(), 1);
  EXPECT_EQ(gb300_4x4.num_hosts_per_partition(), 4);
  EXPECT_EQ(gb300_4x4.num_devices_per_host(), 4);
  EXPECT_TRUE(gb300_4x4.has_gpu_target_config());
  EXPECT_THAT(gb300_4x4.host_target_machine_options(),
              Optional(Property(&cpu::TargetMachineOptions::triple,
                                "aarch64-unknown-linux-gnu")));
}

TEST(GpuTopologyTest, SupportedPlatformConfigsInvariants) {
  absl::Span<const GpuPlatformConfig> configs =
      GetSupportedGpuPlatformConfigs();
  EXPECT_FALSE(configs.empty());
  absl::flat_hash_set<std::string> seen_platform_names;
  absl::flat_hash_set<std::string> seen_short_names;
  absl::flat_hash_set<std::string> seen_device_descriptions;
  for (const GpuPlatformConfig& config : configs) {
    EXPECT_TRUE(seen_platform_names.insert(config.platform_name).second)
        << config.platform_name;
    EXPECT_TRUE(seen_short_names.insert(config.short_name).second)
        << config.short_name;
    ASSERT_FALSE(config.device_descriptions.empty()) << config.platform_name;
    for (const std::string& desc : config.device_descriptions) {
      EXPECT_TRUE(seen_device_descriptions.insert(desc).second) << desc;
    }
    ASSERT_OK_AND_ASSIGN(stream_executor::GpuTargetConfigProto target_config,
                         gpu::GetGpuTargetConfig(config.gpu_model));
    EXPECT_TRUE(absl::c_linear_search(config.device_descriptions,
                                      target_config.device_description_str()))
        << config.platform_name;
    if (config.nvlink_domain_size.has_value()) {
      ASSERT_TRUE(config.devices_per_host.has_value());
      ASSERT_GT(*config.devices_per_host, 0);
      EXPECT_EQ(*config.nvlink_domain_size % *config.devices_per_host, 0);
    }
  }
}

TEST(GpuTopologyTest, GetGpuTopologyForSliceSpec) {
  ASSERT_OK_AND_ASSIGN(GpuTopology h100_multi,
                       GetGpuTopologyForSliceSpec("h100", "8x2"));
  EXPECT_EQ(h100_multi.platform_version(), "nvidia_h100");
  EXPECT_EQ(h100_multi.num_partitions(), 2);
  EXPECT_EQ(h100_multi.num_hosts_per_partition(), 1);
  EXPECT_EQ(h100_multi.num_devices_per_host(), 8);
  EXPECT_EQ(GetGpuDeviceKind(h100_multi.ToProto()), "NVIDIA H100 80GB HBM3");

  ASSERT_OK_AND_ASSIGN(GpuTopology gb300_mnnvl,
                       GetGpuTopologyForSliceSpec("gb300", "4x4"));
  EXPECT_EQ(gb300_mnnvl.platform_version(), "oberon_b300");
  EXPECT_EQ(gb300_mnnvl.num_partitions(), 1);
  EXPECT_EQ(gb300_mnnvl.num_hosts_per_partition(), 4);
  EXPECT_EQ(gb300_mnnvl.num_devices_per_host(), 4);
  EXPECT_EQ(GetGpuDeviceKind(gb300_mnnvl.ToProto()), "NVIDIA GB300");

  ASSERT_OK_AND_ASSIGN(GpuTopology vr200_mnnvl,
                       GetGpuTopologyForSliceSpec("vr200", "4x2"));
  EXPECT_EQ(vr200_mnnvl.platform_version(), "velorum_r200");
  EXPECT_EQ(vr200_mnnvl.num_partitions(), 1);
  EXPECT_EQ(vr200_mnnvl.num_hosts_per_partition(), 2);
  EXPECT_EQ(vr200_mnnvl.num_devices_per_host(), 4);

  EXPECT_THAT(GetGpuTopologyForSliceSpec("h100", "4x2"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetGpuTopologyForSliceSpec("v100", "16x2"),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(GetGpuTopologyForSliceSpec("gb300", "4x20"),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(GpuTopologyTest,
     ParseGpuTopologyPreservesEmbeddedTargetConfigForUnlistedPlatform) {
  const std::string unlisted_with_target_file =
      tsl::io::JoinPath(testing::TempDir(), "unlisted_with_target.txtpb");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(),
                                   unlisted_with_target_file, R"pb(
    platform_version: "custom_gpu"
    num_partitions: 1
    num_hosts_per_partition: 1
    num_devices_per_host: 8
    gpu_target_config { device_description_str: "Custom GPU" }
  )pb"));
  ASSERT_OK_AND_ASSIGN(GpuTopology unlisted_topology,
                       ParseGpuTopology(unlisted_with_target_file));
  EXPECT_EQ(unlisted_topology.platform_version(), "custom_gpu");
  ASSERT_TRUE(unlisted_topology.has_gpu_target_config());
  EXPECT_EQ(unlisted_topology.gpu_target_config().device_description_str,
            "Custom GPU");
  EXPECT_FALSE(unlisted_topology.host_target_machine_options().has_value());

  const std::string registered_alias_with_target_file = tsl::io::JoinPath(
      testing::TempDir(), "registered_alias_with_target.txtpb");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(),
                                   registered_alias_with_target_file, R"pb(
    platform_version: "gb300"
    num_partitions: 1
    num_hosts_per_partition: 1
    num_devices_per_host: 4
    gpu_target_config { device_description_str: "Custom GB300" }
  )pb"));
  ASSERT_OK_AND_ASSIGN(GpuTopology alias_topology,
                       ParseGpuTopology(registered_alias_with_target_file));
  ASSERT_TRUE(alias_topology.has_gpu_target_config());
  EXPECT_EQ(alias_topology.gpu_target_config().device_description_str,
            "Custom GB300");
  EXPECT_THAT(alias_topology.host_target_machine_options(),
              Optional(Property(&cpu::TargetMachineOptions::triple,
                                "aarch64-unknown-linux-gnu")));

  const std::string unlisted_without_target_file =
      tsl::io::JoinPath(testing::TempDir(), "unlisted_without_target.txtpb");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(),
                                   unlisted_without_target_file, R"pb(
    platform_version: "custom_gpu"
    num_partitions: 1
    num_hosts_per_partition: 1
    num_devices_per_host: 8
  )pb"));
  EXPECT_THAT(ParseGpuTopology(unlisted_without_target_file),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla
