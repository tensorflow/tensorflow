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

#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_cliques.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/communicator.h"

namespace xla::gpu {

// Handle to a communicator object with corresponding clique key.
struct CommunicatorHandle {
  CommunicatorHandle(Communicator* comm, GpuCliqueKey clique_key)
      : comm(comm), clique_key(std::move(clique_key)) {}

  Communicator* comm;       // communicator object
  GpuCliqueKey clique_key;  // clique key
};

// Returns a clique key for a collective operation executed for a given set of
// replica groups, group mode and stream kind, based on the `params` argument
// that identifies device that participates in the collective operation.
//
// The `include_participant_groups` argument controls whether the participant
// groups are included in the clique key. Including participant groups is needed
// for safe communicator splitting, as it defines a total order between all
// cliques in the XLA program.
absl::StatusOr<GpuCliqueKey> GetGpuCliqueKey(
    const CollectiveParams& params,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode, AsyncStreamKind stream_kind,
    bool include_participant_groups = true);

// Returns a communicator handle for the given `clique_key` and `params` from
// the set of cliques acquired before the XLA:GPU execution.
absl::StatusOr<CommunicatorHandle> GetComm(
    const CollectiveParams& params, const CollectiveCliques& collective_cliques,
    const GpuCliqueKey& clique_key);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_EXECUTION_H_
