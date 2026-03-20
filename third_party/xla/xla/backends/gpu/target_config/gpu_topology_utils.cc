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

#include "absl/algorithm/container.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {

absl::StatusOr<bool> IsCompatibleWithTargetTopology(
    const GpuTopology& compiler_topology, const GpuTopology& target_topology) {
  // If the platform version is the same, the topologies are compatible.
  if (compiler_topology.platform_version() ==
      target_topology.platform_version()) {
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
  for (const auto& compatible_gpu_pair : kCompatibleGpus) {
    if (absl::c_contains(
            compatible_gpu_pair,
            compiler_topology.gpu_target_config().device_description_str) &&
        absl::c_contains(
            compatible_gpu_pair,
            target_topology.gpu_target_config().device_description_str)) {
      return true;
    }
  }

  return false;
}

}  // namespace xla::gpu
