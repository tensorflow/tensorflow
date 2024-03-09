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
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "xla/pjrt/cpu/cpu_topology.pb.h"

namespace xla {
class CpuTopology {
 public:
  struct CpuDevice {
    int id;
    int process_index;
    int local_hardware_id;

    bool operator==(const CpuDevice& other) const {
      return id == other.id && process_index == other.process_index &&
             local_hardware_id == other.local_hardware_id;
    }
  };

  explicit CpuTopology(std::vector<CpuDevice> cpu_deices)
      : cpu_devices_(std::move(cpu_deices)) {}

  int number_of_devices() const { return cpu_devices_.size(); }
  absl::Span<const CpuDevice> devices() const { return cpu_devices_; }

  static std::unique_ptr<const CpuTopology> FromProto(
      const CpuTopologyProto& proto);
  CpuTopologyProto ToProto() const;

 private:
  const std::vector<CpuDevice> cpu_devices_;
};

}  // namespace xla

#endif  // XLA_PJRT_CPU_CPU_TOPOLOGY_H_
