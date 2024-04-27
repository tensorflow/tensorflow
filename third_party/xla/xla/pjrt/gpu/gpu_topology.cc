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

std::unique_ptr<const GpuTopology> GpuTopology::FromProto(
    const GpuTopologyProto& gpu_topology_proto) {
  return std::make_unique<GpuTopology>(
      std::vector<int>{gpu_topology_proto.device_ids().begin(),
                       gpu_topology_proto.device_ids().end()},
      gpu_topology_proto.platform_version());
}

GpuTopologyProto GpuTopology::ToProto() const {
  GpuTopologyProto proto;
  proto.mutable_device_ids()->Add(device_ids().begin(), device_ids().end());
  proto.set_platform_version(platform_version());
  return proto;
}

}  // namespace xla
