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

#ifndef XLA_PJRT_GPU_GPU_TOPOLOGY_H_
#define XLA_PJRT_GPU_GPU_TOPOLOGY_H_

#include <memory>
#include <vector>

#include "xla/pjrt/gpu/gpu_topology.pb.h"

namespace xla {
class GpuTopology {
 public:
  explicit GpuTopology(const std::vector<int>& gpu_device_ids)
      : devices_ids_(gpu_device_ids) {}

  int number_of_devices() const { return devices_ids_.size(); }
  const std::vector<int>& device_ids() const { return devices_ids_; }

  static std::unique_ptr<const GpuTopology> FromProto(
      const GpuTopologyProto& proto);
  GpuTopologyProto ToProto() const;

 private:
  const std::vector<int> devices_ids_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_TOPOLOGY_H_
