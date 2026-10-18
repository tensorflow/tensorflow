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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_INFO_UTILS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_INFO_UTILS_H_

#include <variant>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/all_gather.h"
#include "xla/backends/gpu/runtime/all_reduce.h"
#include "xla/backends/gpu/runtime/reduce_scatter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu_topology.h"

namespace xla::gpu {

using CollectiveKernelInfo =
    std::variant<AllReduceInfo, AllGatherInfo, ReduceScatterInfo>;

// Builds helper struct for the given collective instruction.
absl::StatusOr<CollectiveKernelInfo> GetCollectiveInfo(
    const HloInstruction* instr, const GpuTopology& gpu_topology,
    const DeviceAssignment* device_assignment = nullptr,
    bool is_multimem_enabled = false);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_INFO_UTILS_H_
