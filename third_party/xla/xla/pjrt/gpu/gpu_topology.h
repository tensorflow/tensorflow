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

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"

namespace xla {
class GpuTopology {
 public:
  struct DefaultTopology {
    const std::vector<int> device_ids;

    bool operator==(const DefaultTopology& other) const {
      return device_ids == other.device_ids;
    }

    int number_of_devices() const { return device_ids.size(); }
    int number_of_hosts() const { return 1; }
  };

  struct PathwaysTopology {
    const int32_t num_slices;
    const int32_t num_hosts_per_slice;
    const int32_t num_devices_per_host;

    bool operator==(const PathwaysTopology& other) const {
      return num_slices == other.num_slices &&
             num_hosts_per_slice == other.num_hosts_per_slice &&
             num_devices_per_host == other.num_devices_per_host;
    }

    int number_of_devices() const {
      return num_slices * num_hosts_per_slice * num_devices_per_host;
    }
    int number_of_hosts() const { return num_slices * num_hosts_per_slice; }
  };

  explicit GpuTopology(const std::vector<int>& gpu_device_ids,
                       absl::string_view platform_version)
      : platform_version_(platform_version),
        topology_kind_(DefaultTopology{gpu_device_ids}) {}
  explicit GpuTopology(absl::string_view platform_version, int32_t num_slices,
                       int32_t num_hosts_per_slice,
                       int32_t num_devices_per_host)
      : platform_version_(platform_version),
        topology_kind_(PathwaysTopology{num_slices, num_hosts_per_slice,
                                        num_devices_per_host}) {}

  bool operator==(const GpuTopology& other) const {
    return platform_version_ == other.platform_version_ &&
           topology_kind_ == other.topology_kind_;
  }
  bool has_default_topology() const {
    return std::holds_alternative<DefaultTopology>(topology_kind_);
  }
  bool has_pathways_topology() const {
    return std::holds_alternative<PathwaysTopology>(topology_kind_);
  }

  int number_of_devices() const;
  const std::vector<int>& device_ids() const {
    return std::get<DefaultTopology>(topology_kind_).device_ids;
  };
  int number_of_hosts() const;

  static std::unique_ptr<const GpuTopology> FromProto(
      const GpuTopologyProto& proto);
  GpuTopologyProto ToProto() const;

  std::string platform_version() const { return platform_version_; }

 private:
  const std::string platform_version_;
  std::variant<DefaultTopology, PathwaysTopology> topology_kind_;
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_TOPOLOGY_H_
