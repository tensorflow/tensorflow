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

#ifndef XLA_PJRT_CPU_CPU_TOPOLOGY_H_
#define XLA_PJRT_CPU_CPU_TOPOLOGY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/pjrt/cpu/cpu_topology.pb.h"
#include "xla/pjrt/pjrt_common.h"

namespace xla {
class CpuTopology {
 public:
  struct CpuDevice {
    int process_id;
    int local_device_id;

    bool operator==(const CpuDevice& other) const {
      return process_id == other.process_id &&
             local_device_id == other.local_device_id;
    }
  };

  explicit CpuTopology(std::vector<CpuDevice> cpu_devices,
                       std::vector<std::string> machine_attributes)
      : cpu_devices_(std::move(cpu_devices)),
        machine_attributes_(std::move(machine_attributes)) {}

  int number_of_devices() const { return cpu_devices_.size(); }
  absl::Span<const CpuDevice> devices() const { return cpu_devices_; }
  absl::Span<const std::string> machine_attributes() const {
    return machine_attributes_;
  }

  static std::unique_ptr<const CpuTopology> FromProto(
      const CpuTopologyProto& proto);
  CpuTopologyProto ToProto() const;

 private:
  const std::vector<CpuDevice> cpu_devices_;
  const std::vector<std::string> machine_attributes_;
};

static const int kMaxCpuDevicesPerProcess = 1 << 17;

inline PjRtGlobalDeviceId PackCpuDeviceId(int process_index, int device_id) {
  return PjRtGlobalDeviceId(kMaxCpuDevicesPerProcess * process_index +
                            device_id);
}

inline int UnpackCpuProcessIndex(PjRtGlobalDeviceId global_device_id) {
  return global_device_id.value() / kMaxCpuDevicesPerProcess;
}

inline int UnpackCpuDeviceId(PjRtGlobalDeviceId global_device_id) {
  return global_device_id.value() % kMaxCpuDevicesPerProcess;
}

}  // namespace xla

#endif  // XLA_PJRT_CPU_CPU_TOPOLOGY_H_
