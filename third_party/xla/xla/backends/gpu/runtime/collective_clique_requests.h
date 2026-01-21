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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUE_REQUESTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUE_REQUESTS_H_

#include <cstddef>
#include <optional>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"

namespace xla::gpu {

// Collective thunks (including collective FFI calls) can request communicators
// for various collective cliques. XLA runtime is responsible for collecting
// such requests during the prepare stage and acquiring the cliques during the
// initialize stage.
class CollectiveCliqueRequests {
 public:
  // For each requested clique key, we also assign a monotonically increasing
  // id, that allows us to deterministically order clique requests.
  //
  // Example: 8 ranks splitted in different groups of communicators
  //
  // Group #0: [0,1], [2,3], [4,5], [6,7]
  // Group #1: [0,4], [1,5], [2,6], [3,7]
  //
  // Both groups #0 and #1 can be acqured by splitting [0...7] clique. To avoid
  // deadlocks all participants should acquire all cliques in a group #0 before
  // acquiring any cliques in a group #1.
  //
  // We rely on clique request id to guarantee that the order is identical
  // on all participating ranks (including ranks running on different hosts).
  //
  // Remember that clique requests are collected independently by running thunk
  // sequence prepare stage in parallel for all ranks. After all requests are
  // collected, XLA runtime initializes communicators for all requested cliques.
  //
  // This initialization must happen in identical order across all ranks, and
  // ranks might be running as separate processes or even on separate hosts, so
  // any communication between ranks is impossible.
  //
  // We rely on the fact, that XLA uses SPMD programming model, and all ranks
  // execute identical thunk sequence in exact same order, and assigned request
  // id is essentially a thunk index in the parent thunk sequence. However we
  // don't want to sort just by request index, because acquiring large cliques
  // first improves performance and memory utilization, as we have more chances
  // to reuse existing communicators when requesting smaller cliques via
  // communicator splitting.
  struct CliqueRequest {
    size_t id;
    GpuCliqueKey key;

    // Requirements for device communicators. We keep them in a sorted container
    // to guarantee that all ranks create device communicators in the same
    // order, otherwise it might lead to deadlocks.
    absl::btree_set<GpuDeviceCommunicator::Requirements> dev_comms;
  };

  // An extra set of requirements for the collective clique. When XLA runtime
  // acquires collective cliques (see `collective_cliques.h`), it will satisfy
  // all of the requirements, or will return an error.
  struct CliqueRequirements {
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const CliqueRequirements& reqs) {
      if (reqs.dev_comm) {
        absl::Format(&sink, "{dev_comm: %v}", *reqs.dev_comm);
      } else {
        absl::Format(&sink, "{dev_comm: n/a}");
      }
    }

    // Create a device communicator for the given collective clique.
    std::optional<GpuDeviceCommunicator::Requirements> dev_comm;
  };

  // Adds a clique key to the list of requested cliques.
  absl::Status RequestClique(const GpuCliqueKey& clique_key,
                             const CliqueRequirements& requirements = {});

  // Returns all requested cliques in undefined order.
  std::vector<GpuCliqueKey> RequestedCliques() const;

  // Returns all requested cliques in a deterministic order optimized for
  // efficient communicator acquisition.
  std::vector<CliqueRequest> OrderedRequestedCliques() const;

 private:
  absl::flat_hash_map<GpuCliqueKey, CliqueRequest> cliques_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUE_REQUESTS_H_
