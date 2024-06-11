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

#include "absl/strings/string_view.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"

namespace xla {
class GpuTopology {
 public:
  explicit GpuTopology(const std::vector<int>& gpu_device_ids,
                       absl::string_view platform_version, int32_t num_slices,
                       int32_t num_hosts_per_slice,
                       int32_t num_devices_per_host)
      : devices_ids_(gpu_device_ids),
        platform_version_(platform_version),
        num_slices_(num_slices),
        num_hosts_per_slice_(num_hosts_per_slice),
        num_devices_per_host_(num_devices_per_host) {}

  bool operator==(const GpuTopology& other) const {
    return devices_ids_ == other.devices_ids_ &&
           platform_version_ == other.platform_version_ &&
           num_slices_ == other.num_slices_ &&
           num_hosts_per_slice_ == other.num_hosts_per_slice_ &&
           num_devices_per_host_ == other.num_devices_per_host_;
  }

  int number_of_devices() const {
    return number_of_hosts() * num_devices_per_host_;
  }
  const std::vector<int>& device_ids() const { return devices_ids_; }
  int number_of_hosts() const { return num_slices_ * num_hosts_per_slice_; }

  static std::unique_ptr<const GpuTopology> FromProto(
      const GpuTopologyProto& proto);
  GpuTopologyProto ToProto() const;

  std::string platform_version() const { return platform_version_; }
  int32_t num_slices() const { return num_slices_; }
  int32_t num_hosts_per_slice() const { return num_hosts_per_slice_; }
  int32_t num_devices_per_host() const { return num_devices_per_host_; }

 private:
  const std::vector<int> devices_ids_;
  const std::string platform_version_;
  const int32_t num_slices_;
  const int32_t num_hosts_per_slice_;
  const int32_t num_devices_per_host_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_TOPOLOGY_H_
