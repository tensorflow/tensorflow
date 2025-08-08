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

#include "absl/strings/string_view.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"

namespace xla {
class GpuTopology {
 public:
  explicit GpuTopology(absl::string_view platform_version,
                       int32_t num_partitions, int32_t num_hosts_per_partition,
                       int32_t num_devices_per_host)
      : platform_version_(platform_version),
        num_partitions_(num_partitions),
        num_hosts_per_partition_(num_hosts_per_partition),
        num_devices_per_host_(num_devices_per_host) {}

  bool operator==(const GpuTopology& other) const {
    return platform_version_ == other.platform_version_ &&
           num_partitions_ == other.num_partitions_ &&
           num_hosts_per_partition_ == other.num_hosts_per_partition_ &&
           num_devices_per_host_ == other.num_devices_per_host_;
  }

  int number_of_devices() const {
    return is_topology_symmetric() ? number_of_hosts() * num_devices_per_host_
                                   : -1;
  }

  int number_of_hosts() const {
    return is_topology_symmetric() ? num_partitions_ * num_hosts_per_partition_
                                   : -1;
  }

  static std::unique_ptr<const GpuTopology> FromProto(
      const GpuTopologyProto& proto);
  GpuTopologyProto ToProto() const;

  absl::string_view platform_version() const { return platform_version_; }
  int32_t num_partitions() const { return num_partitions_; }
  int32_t num_hosts_per_partition() const { return num_hosts_per_partition_; }
  int32_t num_devices_per_host() const { return num_devices_per_host_; }

 private:
  const std::string platform_version_;
  const int32_t num_partitions_;
  const int32_t num_hosts_per_partition_;
  const int32_t num_devices_per_host_;

  bool is_topology_symmetric() const {
    return num_partitions_ != -1 && num_hosts_per_partition_ != -1 &&
           num_devices_per_host_ != -1;
  }
};

}  // namespace xla

#endif  // XLA_PJRT_GPU_GPU_TOPOLOGY_H_
