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

#ifndef XLA_SERVICE_GPU_TOPOLOGY_H_
#define XLA_SERVICE_GPU_TOPOLOGY_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/gpu_topology.pb.h"

namespace xla {

// System-level configuration for a known GPU platform, combining the target
// chip model with default host and NVLink domain geometry and known runtime
// device description strings.
struct GpuPlatformConfig {
  std::string platform_name;
  std::string short_name;
  gpu::GpuModel gpu_model;
  std::optional<int32_t> devices_per_host;
  std::optional<int32_t> nvlink_domain_size;
  std::vector<std::string> device_descriptions;
};

class GpuTopology {
 public:
  explicit GpuTopology(
      absl::string_view platform_version, int32_t num_partitions,
      int32_t num_hosts_per_partition, int32_t num_devices_per_host,
      std::optional<gpu::GpuTargetConfig> gpu_target_config = std::nullopt,
      std::optional<cpu::TargetMachineOptions> host_target_machine_options =
          std::nullopt,
      std::optional<int32_t> num_devices_per_process = std::nullopt)
      : platform_version_(platform_version),
        num_partitions_(num_partitions),
        num_hosts_per_partition_(num_hosts_per_partition),
        num_devices_per_host_(num_devices_per_host),
        gpu_target_config_(std::move(gpu_target_config)),
        host_target_machine_options_(std::move(host_target_machine_options)),
        // Assume single process per host if not specified.
        num_devices_per_process_(num_devices_per_process) {}

  // Returns a copy of this topology with the GpuTargetConfig field set to
  // `gpu_target_config`.
  GpuTopology CopyWithNewTargetConfig(
      const gpu::GpuTargetConfig& gpu_target_config) const {
    GpuTopology ret(*this);
    ret.gpu_target_config_ = gpu_target_config;
    return ret;
  }

  bool operator==(const GpuTopology& other) const {
    return platform_version_ == other.platform_version_ &&
           num_partitions_ == other.num_partitions_ &&
           num_hosts_per_partition_ == other.num_hosts_per_partition_ &&
           num_devices_per_host_ == other.num_devices_per_host_ &&
           gpu_target_config_ == other.gpu_target_config_ &&
           host_target_machine_options_ == other.host_target_machine_options_;
  }

  bool operator!=(const GpuTopology& other) const { return !(*this == other); }

  int number_of_devices() const {
    return is_topology_symmetric() ? number_of_hosts() * num_devices_per_host_
                                   : -1;
  }

  int number_of_hosts() const {
    return is_topology_symmetric() ? num_partitions_ * num_hosts_per_partition_
                                   : -1;
  }

  static absl::StatusOr<std::unique_ptr<const GpuTopology>> FromProto(
      const GpuTopologyProto& proto);
  GpuTopologyProto ToProto() const;

  absl::string_view platform_version() const { return platform_version_; }
  int32_t num_partitions() const { return num_partitions_; }
  int32_t num_hosts_per_partition() const { return num_hosts_per_partition_; }
  int32_t num_devices_per_host() const { return num_devices_per_host_; }
  // Returns the number of devices per process. If not set, returns the number
  // of devices per host.
  int32_t num_devices_per_process() const {
    return num_devices_per_process_.value_or(num_devices_per_host_);
  }
  // Returns the number of devices per host as loaded from the proto.
  std::optional<int32_t> num_devices_per_process_opt() const {
    return num_devices_per_process_;
  }
  bool is_single_process_per_host() const {
    return num_devices_per_process() == num_devices_per_host_;
  }
  int32_t slice_size() const {
    return num_hosts_per_partition() * num_devices_per_host();
  }

  bool has_gpu_target_config() const { return gpu_target_config_.has_value(); }
  const gpu::GpuTargetConfig& gpu_target_config() const {
    return *gpu_target_config_;
  }
  int32_t locality_domain_count() const {
    return gpu_target_config_.has_value()
               ? gpu_target_config_->device_description.locality_domain_count()
               : 0;
  }

  const std::optional<cpu::TargetMachineOptions>& host_target_machine_options()
      const {
    return host_target_machine_options_;
  }

  bool confidential_computing_enabled() const {
    if (has_gpu_target_config()) {
      return gpu_target_config()
          .device_description.confidential_computing_enabled();
    }
    return false;
  }

 private:
  std::string platform_version_;
  int32_t num_partitions_;
  int32_t num_hosts_per_partition_;
  int32_t num_devices_per_host_;
  std::optional<gpu::GpuTargetConfig> gpu_target_config_;
  std::optional<cpu::TargetMachineOptions> host_target_machine_options_;
  std::optional<int32_t> num_devices_per_process_;

  bool is_topology_symmetric() const {
    return num_partitions_ != -1 && num_hosts_per_partition_ != -1 &&
           num_devices_per_host_ != -1;
  }
};

// Returns all registered GPU platform configurations.
absl::Span<const GpuPlatformConfig> GetSupportedGpuPlatformConfigs();

// Looks up a `GpuPlatformConfig` by canonical platform name (e.g.,
// "nvidia_h100", "oberon_b300"), short name (e.g., "h100", "gb300"), or
// embedded device description string (e.g., "NVIDIA H100 80GB HBM3").
// The returned pointer refers to an entry in the static registry returned by
// `GetSupportedGpuPlatformConfigs()` and remains valid for the program
// lifetime.
absl::StatusOr<const GpuPlatformConfig*> GetGpuPlatformConfig(
    absl::string_view platform_or_device_name);

absl::StatusOr<GpuTopology> GetGpuTopologyForPlatform(
    absl::string_view platform_version, int32_t num_partitions,
    int32_t num_hosts_per_partition, int32_t num_devices_per_host);

// Constructs a `GpuTopology` for `platform_or_short_name` and a slice
// specification of the form "<num_devices>" (single-host) or
// "<devices_per_host>x<num_hosts>" (multi-host), validating multi-host
// `devices_per_host` and deriving `(num_partitions, num_hosts_per_partition)`
// from the platform's NVLink domain size.
absl::StatusOr<GpuTopology> GetGpuTopologyForSliceSpec(
    absl::string_view platform_or_short_name, absl::string_view slice_spec);

// Returns the human-readable device kind string (e.g.,
// "NVIDIA H100 80GB HBM3") for `gpu_topology_proto`.
std::string GetGpuDeviceKind(const GpuTopologyProto& gpu_topology_proto);

GpuTopology GetSingleDeviceGpuTopology(
    absl::string_view platform_version,
    const gpu::GpuTargetConfig& gpu_target_config,
    const std::optional<cpu::TargetMachineOptions>&
        host_target_machine_options = std::nullopt);

// Parses a `GpuTopology` from either an inline topology specification
// (`[<platform_version>:]<num_partitions>x<num_hosts_per_partition>x<num_devices_per_host>`
// or `[<platform_version>:]<num_hosts_per_partition>x<num_devices_per_host>`)
// or a `GpuTopologyProto` textproto file at `topology_spec_or_filename`.
absl::StatusOr<GpuTopology> ParseGpuTopology(
    absl::string_view topology_spec_or_filename);

}  // namespace xla

#endif  // XLA_SERVICE_GPU_TOPOLOGY_H_
