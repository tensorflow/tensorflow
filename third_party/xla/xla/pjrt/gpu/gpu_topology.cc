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
#include <string>
#include <vector>

namespace xla {

std::unique_ptr<const GpuTopology> GpuTopology::FromProto(
    const GpuTopologyProto& gpu_topology_proto) {
  return std::make_unique<GpuTopology>(
      std::vector<int>{gpu_topology_proto.device_ids().begin(),
                       gpu_topology_proto.device_ids().end()},
      gpu_topology_proto.platform_version(), gpu_topology_proto.num_slices(),
      gpu_topology_proto.num_hosts_per_slice(),
      gpu_topology_proto.num_devices_per_host());
}

GpuTopologyProto GpuTopology::ToProto() const {
  GpuTopologyProto proto;
  proto.mutable_device_ids()->Add(device_ids().begin(), device_ids().end());
  proto.set_platform_version(std::string(platform_version()));
  proto.set_num_slices(num_slices());
  proto.set_num_hosts_per_slice(num_hosts_per_slice());
  proto.set_num_devices_per_host(num_devices_per_host());
  return proto;
}

}  // namespace xla
