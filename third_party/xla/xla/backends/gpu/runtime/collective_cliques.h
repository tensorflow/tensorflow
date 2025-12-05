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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUES_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUES_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla::gpu {

// A collection of collective cliques acquired based on GPU clique requests
// collected from all thunks at prepare stage.
class CollectiveCliques {
 public:
  CollectiveCliques() = default;
  CollectiveCliques(AcquiredCliquesMap cliques_map,
                    int32_t num_transient_cliques);

  absl::StatusOr<Communicator*> GetComm(const GpuCliqueKey& clique_key,
                                        RankId rank) const;

  // Returns whether peer device memory access is possible between all devices
  // in the clique.
  absl::StatusOr<bool> peer_access_enabled(
      const GpuCliqueKey& clique_key) const;

  bool empty() const { return cliques_map_.empty(); }

  bool num_transient_cliques() const { return num_transient_cliques_; }

 private:
  AcquiredCliquesMap cliques_map_;

  // The number of acquired non-persistent clique. We need to keep track of
  // newly created communicators to insert rendezvous after first
  // initialization, because otherwise we observe deadlocks with NCCL
  // collectives backends.
  int32_t num_transient_cliques_ = 0;
};

// Acquires collective cliques using the given collective parameters for all
// requested GPU cliques.
absl::StatusOr<CollectiveCliques> AcquireCollectiveCliques(
    const CollectiveParams& params, const CollectiveCliqueRequests& cliques,
    bool use_persistent_cliques);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUES_H_
