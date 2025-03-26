/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RESOURCE_REQUESTS_H_
#define XLA_SERVICE_GPU_RESOURCE_REQUESTS_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/thunk.h"

namespace xla {
namespace gpu {

// Shared resources required for thunk initialization and execution.
class ResourceRequests : public Thunk::ResourceRequestsInterface {
 public:
  absl::Status AddClique(const GpuCliqueKey& clique_key) final;

  absl::StatusOr<Thunk::CollectiveCliques> AcquireCollectiveCliques(
      const Thunk::CollectiveExecuteParams& params,
      bool use_persistent_cliques);

 private:
  struct CliqueRequest {
    GpuCliqueKey key;
    int64_t id;
  };

  // Return clique requests deterministically ordered using a comparison
  // function that produces identical ordering for all participating ranks.
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
  std::vector<CliqueRequest> GetOrderedCliqueRequests();

  absl::flat_hash_map<GpuCliqueKey, CliqueRequest> cliques_;
};
}  // namespace gpu
}  // namespace xla
#endif  // XLA_SERVICE_GPU_RESOURCE_REQUESTS_H_
