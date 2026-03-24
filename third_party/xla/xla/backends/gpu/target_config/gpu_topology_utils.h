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

#ifndef XLA_BACKENDS_GPU_TARGET_CONFIG_GPU_TOPOLOGY_UTILS_H_
#define XLA_BACKENDS_GPU_TARGET_CONFIG_GPU_TOPOLOGY_UTILS_H_

#include "absl/status/statusor.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {

// Returns true if a host with `compiler_topology` can be used to compile for a
// host with `target_topology`.
absl::StatusOr<bool> IsCompatibleWithTargetTopology(
    const GpuTopology& compiler_topology, const GpuTopology& target_topology);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TARGET_CONFIG_GPU_TOPOLOGY_UTILS_H_
