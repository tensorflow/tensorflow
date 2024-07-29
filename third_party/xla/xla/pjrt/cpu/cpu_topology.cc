/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/pjrt/cpu/cpu_topology.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace xla {

std::unique_ptr<const CpuTopology> CpuTopology::FromProto(
    const CpuTopologyProto& cpu_topology_proto) {
  std::vector<CpuTopology::CpuDevice> devices;
  devices.reserve(cpu_topology_proto.cpu_devices_size());

  for (size_t i = 0; i < cpu_topology_proto.cpu_devices_size(); ++i) {
    auto& cpu_device_proto = cpu_topology_proto.cpu_devices(i);
    devices.push_back(CpuDevice{cpu_device_proto.process_index(),
                                cpu_device_proto.local_hardware_id()});
  }

  std::vector<std::string> machine_attributes;
  machine_attributes.reserve(cpu_topology_proto.machine_attributes_size());
  for (size_t i = 0; i < cpu_topology_proto.machine_attributes_size(); ++i) {
    machine_attributes.push_back(cpu_topology_proto.machine_attributes(i));
  }

  return std::make_unique<CpuTopology>(std::move(devices),
                                       std::move(machine_attributes));
}

CpuTopologyProto CpuTopology::ToProto() const {
  CpuTopologyProto proto;
  for (auto& cpu_device : cpu_devices_) {
    auto* cpu_device_proto = proto.add_cpu_devices();
    cpu_device_proto->set_process_index(cpu_device.process_id);
    cpu_device_proto->set_local_hardware_id(cpu_device.local_device_id);
  }
  for (const std::string& machine_attribute : machine_attributes_) {
    proto.add_machine_attributes(machine_attribute);
  }
  return proto;
}

}  // namespace xla
