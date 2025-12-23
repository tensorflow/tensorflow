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

#include "xla/pjrt/gpu/gpu_topology.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/service/compiler.h"

namespace xla {

std::unique_ptr<const GpuTopology> GpuTopology::FromProto(
    const GpuTopologyProto& gpu_topology_proto) {
  std::optional<Compiler::GpuTargetConfig> gpu_target_config = std::nullopt;
  if (gpu_topology_proto.has_gpu_target_config()) {
    auto gpu_target_config_or = Compiler::GpuTargetConfig::FromProto(
        gpu_topology_proto.gpu_target_config());
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

}  // namespace xla
