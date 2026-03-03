/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_EXECUTION_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_EXECUTION_H_

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {

// Returns a clique key for a collective operation executed for a given set of
// replica groups, group mode and stream kind, based on the `params` argument
// that identifies device that participates in the collective operation.
absl::StatusOr<GpuCliqueKey> GetGpuCliqueKey(
    const CollectiveParams& params,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode, bool is_p2p);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_EXECUTION_H_
