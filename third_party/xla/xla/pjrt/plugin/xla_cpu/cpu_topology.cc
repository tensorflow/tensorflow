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

#include "xla/pjrt/plugin/xla_cpu/cpu_topology.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {

absl::StatusOr<std::unique_ptr<const CpuTopology>> CpuTopology::FromProto(
    const CpuTopologyProto& cpu_topology_proto) {
  std::vector<CpuTopology::CpuDevice> devices;
  devices.reserve(cpu_topology_proto.cpu_devices_size());

  for (size_t i = 0; i < cpu_topology_proto.cpu_devices_size(); ++i) {
    auto& cpu_device_proto = cpu_topology_proto.cpu_devices(i);
    devices.push_back(CpuDevice{cpu_device_proto.process_index(),
                                cpu_device_proto.local_hardware_id()});
  }

  ASSIGN_OR_RETURN(auto target_machine_options,
                   cpu::TargetMachineOptions::FromProto(
                       cpu_topology_proto.target_machine_options()));

  return std::make_unique<CpuTopology>(std::move(devices),
                                       std::move(target_machine_options));
}

CpuTopologyProto CpuTopology::ToProto() const {
  CpuTopologyProto proto;
  for (auto& cpu_device : cpu_devices_) {
    auto* cpu_device_proto = proto.add_cpu_devices();
    cpu_device_proto->set_process_index(cpu_device.process_id);
    cpu_device_proto->set_local_hardware_id(cpu_device.local_device_id);
  }
  *proto.mutable_target_machine_options() = target_machine_options_.ToProto();
  return proto;
}

}  // namespace xla
