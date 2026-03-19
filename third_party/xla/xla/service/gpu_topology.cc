/* Copyright 2023 The OpenXLA Authors.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace {

absl::StatusOr<gpu::GpuModel> GetGpuModel(absl::string_view platform_type) {
  if (platform_type == "tesla_a100") {
    return gpu::GpuModel::A100_SXM_40;
  }
  if (platform_type == "nvidia_h100") {
    return gpu::GpuModel::H100_SXM;
  }
  if (platform_type == "umbriel_b200" || platform_type == "oberon_b200") {
    return gpu::GpuModel::B200;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported GPU platform type: ", platform_type));
}

}  // namespace

std::unique_ptr<const GpuTopology> GpuTopology::FromProto(
    const GpuTopologyProto& gpu_topology_proto) {
  std::optional<gpu::GpuTargetConfig> gpu_target_config = std::nullopt;
  if (gpu_topology_proto.has_gpu_target_config()) {
    auto gpu_target_config_or =
        gpu::GpuTargetConfig::FromProto(gpu_topology_proto.gpu_target_config());
    CHECK_OK(gpu_target_config_or);
    gpu_target_config = *std::move(gpu_target_config_or);
  }
  return std::make_unique<GpuTopology>(
      gpu_topology_proto.platform_version(),
      gpu_topology_proto.num_partitions(),
      gpu_topology_proto.num_hosts_per_partition(),
      gpu_topology_proto.num_devices_per_host(), std::move(gpu_target_config));
}

GpuTopologyProto GpuTopology::ToProto() const {
  GpuTopologyProto proto;
  proto.set_platform_version(platform_version());
  proto.set_num_partitions(num_partitions());
  proto.set_num_hosts_per_partition(num_hosts_per_partition());
  proto.set_num_devices_per_host(num_devices_per_host());
  if (gpu_target_config_.has_value()) {
    *proto.mutable_gpu_target_config() = gpu_target_config().ToProto();
  }
  return proto;
}

absl::StatusOr<GpuTopology> GetGpuTopologyForPlatform(
    absl::string_view platform_version, int32_t num_partitions,
    int32_t num_hosts_per_partition, int32_t num_devices_per_host) {
  // TODO(b/470487616): Don't use string matching to get the GpuTargetConfig.
  ASSIGN_OR_RETURN(auto spec_name, GetGpuModel(platform_version));
  ASSIGN_OR_RETURN(auto gpu_target_config_proto,
                   gpu::GetGpuTargetConfig(spec_name));
  ASSIGN_OR_RETURN(auto gpu_target_config,
                   gpu::GpuTargetConfig::FromProto(gpu_target_config_proto));
  return GpuTopology(platform_version, num_partitions, num_hosts_per_partition,
                     num_devices_per_host, std::move(gpu_target_config));
}

GpuTopology GetSingleDeviceGpuTopology(
    absl::string_view platform_version,
    const gpu::GpuTargetConfig& gpu_target_config) {
  return GpuTopology(platform_version, 1, 1, 1, gpu_target_config);
}

}  // namespace xla
