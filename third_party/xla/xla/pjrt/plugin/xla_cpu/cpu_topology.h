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

#ifndef XLA_PJRT_PLUGIN_XLA_CPU_CPU_TOPOLOGY_H_
#define XLA_PJRT_PLUGIN_XLA_CPU_CPU_TOPOLOGY_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_topology.pb.h"

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
                       xla::cpu::TargetMachineOptions target_machine_options)
      : cpu_devices_(std::move(cpu_devices)),
        target_machine_options_(std::move(target_machine_options)) {}

  int number_of_devices() const { return cpu_devices_.size(); }
  absl::Span<const CpuDevice> devices() const { return cpu_devices_; }
  const xla::cpu::TargetMachineOptions& target_machine_options() const {
    return target_machine_options_;
  }

  static absl::StatusOr<std::unique_ptr<const CpuTopology>> FromProto(
      const CpuTopologyProto& proto);
  CpuTopologyProto ToProto() const;

  bool operator==(const CpuTopology& other) const {
    return cpu_devices_ == other.cpu_devices_ &&
           target_machine_options_ == other.target_machine_options_;
  }

 private:
  const std::vector<CpuDevice> cpu_devices_;
  const xla::cpu::TargetMachineOptions target_machine_options_;
};

static const int kMaxCpuDevicesPerProcess = 1 << 11;

inline GlobalDeviceId PackCpuDeviceId(int process_index, int device_id) {
  return GlobalDeviceId(kMaxCpuDevicesPerProcess * process_index + device_id);
}

inline int UnpackCpuProcessIndex(GlobalDeviceId global_device_id) {
  return global_device_id.value() / kMaxCpuDevicesPerProcess;
}

inline int UnpackCpuLocalDeviceId(GlobalDeviceId global_device_id) {
  return global_device_id.value() % kMaxCpuDevicesPerProcess;
}

}  // namespace xla

#endif  // XLA_PJRT_PLUGIN_XLA_CPU_CPU_TOPOLOGY_H_
