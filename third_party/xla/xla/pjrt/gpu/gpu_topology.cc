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
#include <vector>

namespace xla {
int GpuTopology::number_of_devices() const {
  if (has_default_topology()) {
    return std::get<DefaultTopology>(topology_kind_).number_of_devices();
  }
  return std::get<PathwaysTopology>(topology_kind_).number_of_devices();
}

int GpuTopology::number_of_hosts() const {
  if (has_default_topology()) {
    return std::get<DefaultTopology>(topology_kind_).number_of_hosts();
  }
  return std::get<PathwaysTopology>(topology_kind_).number_of_hosts();
}

std::unique_ptr<const GpuTopology> GpuTopology::FromProto(
    const GpuTopologyProto& gpu_topology_proto) {
  if (gpu_topology_proto.topology_kind_case() ==
      GpuTopologyProto::kDefaultTopology) {
    return std::make_unique<GpuTopology>(
        std::vector<int>(
            gpu_topology_proto.default_topology().device_ids().begin(),
            gpu_topology_proto.default_topology().device_ids().end()),
        gpu_topology_proto.platform_version());
  }

  return std::make_unique<GpuTopology>(
      gpu_topology_proto.platform_version(),
      gpu_topology_proto.pathways_topology().num_slices(),
      gpu_topology_proto.pathways_topology().num_hosts_per_slice(),
      gpu_topology_proto.pathways_topology().num_devices_per_host());
}

GpuTopologyProto GpuTopology::ToProto() const {
  GpuTopologyProto proto;
  proto.set_platform_version(platform_version());
  if (has_default_topology()) {
    DefaultTopology default_topology =
        std::get<DefaultTopology>(topology_kind_);
    proto.mutable_default_topology()->mutable_device_ids()->Add(
        default_topology.device_ids.begin(), default_topology.device_ids.end());
  } else if (has_pathways_topology()) {
    PathwaysTopology pathways_topology =
        std::get<PathwaysTopology>(topology_kind_);
    proto.mutable_pathways_topology()->set_num_slices(
        pathways_topology.num_slices);
    proto.mutable_pathways_topology()->set_num_hosts_per_slice(
        pathways_topology.num_hosts_per_slice);
    proto.mutable_pathways_topology()->set_num_devices_per_host(
        pathways_topology.num_devices_per_host);
  }
  return proto;
}

}  // namespace xla
