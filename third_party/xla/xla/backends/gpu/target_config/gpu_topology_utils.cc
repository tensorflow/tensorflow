/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/target_config/gpu_topology_utils.h"

#include <array>
#include <memory>
#include <string>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/target_config/target_config.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {
namespace {

// Returns the primary non-empty device or platform identifier from `topology`,
// checking `gpu_target_config().device_description_str`,
// `gpu_target_config().device_description.name()`, and `platform_version()` in
// order.
absl::string_view PrimaryTopologyDeviceIdentifier(const GpuTopology& topology) {
  if (topology.has_gpu_target_config()) {
    if (!topology.gpu_target_config().device_description_str.empty()) {
      return topology.gpu_target_config().device_description_str;
    }
    if (!topology.gpu_target_config().device_description.name().empty()) {
      return topology.gpu_target_config().device_description.name();
    }
  }
  return topology.platform_version();
}

// Resolves a canonical device description string for `topology`, normalizing
// SKU variants belonging to a registered `GpuPlatformConfig` to that platform's
// primary device description string.
absl::StatusOr<std::string> ResolveDeviceDescriptionStr(
    const GpuTopology& topology) {
  if (topology.has_gpu_target_config()) {
    for (absl::string_view candidate :
         {absl::string_view(
              topology.gpu_target_config().device_description_str),
          absl::string_view(
              topology.gpu_target_config().device_description.name())}) {
      if (!candidate.empty()) {
        absl::StatusOr<const GpuPlatformConfig*> config =
            GetGpuPlatformConfig(candidate);
        if (config.ok() && !(*config)->device_descriptions.empty()) {
          return (*config)->device_descriptions.front();
        }
        return std::string(candidate);
      }
    }
  }
  if (!topology.platform_version().empty()) {
    absl::StatusOr<const GpuPlatformConfig*> config =
        GetGpuPlatformConfig(topology.platform_version());
    if (config.ok() && !(*config)->device_descriptions.empty()) {
      return (*config)->device_descriptions.front();
    }
  }
  return absl::InvalidArgumentError(
      "GpuTargetConfig is required to determine compatibility.");
}

}  // namespace

absl::Status ValidateSupportedGpuTopology(const GpuTopology& gpu_topology) {
  absl::string_view identifier = PrimaryTopologyDeviceIdentifier(gpu_topology);
  if (identifier.empty()) {
    return absl::InvalidArgumentError(
        "GPU topology does not specify a device or platform identifier.");
  }
  return GetGpuPlatformConfig(identifier).status();
}

absl::Status ValidateSupportedGpuTopology(
    const xla::GpuTopologyProto& gpu_topology_proto) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<const GpuTopology> gpu_topology,
                   GpuTopology::FromProto(gpu_topology_proto));
  return ValidateSupportedGpuTopology(*gpu_topology);
}

absl::StatusOr<bool> IsCompatibleWithTargetTopology(
    const GpuTopology& compiler_topology, const GpuTopology& target_topology) {
  // A compiler topology with no locality-domain capability does not constrain
  // the target. Once it records multiple domains, the target must match so
  // encoded per-domain memory-space colors retain their meaning.
  if (compiler_topology.locality_domain_count() > 1 &&
      compiler_topology.locality_domain_count() !=
          target_topology.locality_domain_count()) {
    return false;
  }

  // If a non-empty platform version is the same, the topologies are compatible.
  if (!compiler_topology.platform_version().empty() &&
      compiler_topology.platform_version() ==
          target_topology.platform_version()) {
    return true;
  }

  ABSL_ASSIGN_OR_RETURN(std::string compiler_device_description,
                   ResolveDeviceDescriptionStr(compiler_topology));
  ABSL_ASSIGN_OR_RETURN(std::string target_device_description,
                   ResolveDeviceDescriptionStr(target_topology));

  if (!compiler_device_description.empty() &&
      compiler_device_description == target_device_description) {
    return true;
  }

  using CompatibleGpuPair = std::array<absl::string_view, 2>;
  constexpr std::array kCompatibleGpus = {
      // NVIDIA B200 (HGX) and B200 (Superchip) are similar enough to consider
      // them compatible.
      CompatibleGpuPair{"NVIDIA B200", "NVIDIA GB200"},

      // NVIDIA B300 (HGX) and B300 (Superchip) are similar enough to consider
      // them compatible.
      CompatibleGpuPair{"NVIDIA GB300", "NVIDIA B300"}};
  for (const CompatibleGpuPair& compatible_gpu_pair : kCompatibleGpus) {
    if (absl::c_contains(compatible_gpu_pair, compiler_device_description) &&
        absl::c_contains(compatible_gpu_pair, target_device_description)) {
      return true;
    }
  }

  return false;
}

absl::StatusOr<bool> IsCompatibleWithTargetTopology(
    const xla::GpuTopologyProto& compiler_topology_proto,
    const xla::GpuTopologyProto& target_topology_proto) {
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<const GpuTopology> compiler_topology,
                   GpuTopology::FromProto(compiler_topology_proto));
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<const GpuTopology> target_topology,
                   GpuTopology::FromProto(target_topology_proto));
  return IsCompatibleWithTargetTopology(*compiler_topology, *target_topology);
}

}  // namespace xla::gpu
