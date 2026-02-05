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

#include "xla/backends/gpu/runtime/collective_clique_requests.h"

#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"

namespace xla::gpu {

absl::Status CollectiveCliqueRequests::RequestClique(
    const GpuCliqueKey& clique_key, const CliqueRequirements& requirements) {
  VLOG(5) << "Add collective clique request: " << clique_key.ToString()
          << "; requirements=" << requirements;

  // If the clique already exist, update it with new requirementes.
  if (auto it = cliques_.find(clique_key); it != cliques_.end()) {
    CliqueRequest& req = it->second;
    if (requirements.dev_comm) {
      req.dev_comms.insert(*requirements.dev_comm);
    }
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // clique requests simply based on the number of already recorded requests.
  CliqueRequest req{/*id=*/cliques_.size(), clique_key};
  if (requirements.dev_comm) {
    req.dev_comms.insert(*requirements.dev_comm);
  }

  cliques_.try_emplace(clique_key, std::move(req));
  return absl::OkStatus();
}

std::vector<GpuCliqueKey> CollectiveCliqueRequests::RequestedCliques() const {
  std::vector<GpuCliqueKey> clique_keys;
  clique_keys.reserve(cliques_.size());
  for (const auto& [key, _] : cliques_) {
    clique_keys.push_back(key);
  }

  return clique_keys;
}

std::vector<CollectiveCliqueRequests::CliqueRequest>
CollectiveCliqueRequests::OrderedRequestedCliques() const {
  std::vector<CliqueRequest> cliques;
  cliques.reserve(cliques_.size());
  for (const auto& [_, request] : cliques_) {
    cliques.push_back(request);
  }

  absl::c_sort(cliques, [](const CliqueRequest& a, const CliqueRequest& b) {
    // Acquire larger cliques first to be able to split them later.
    if (a.key.devices().size() > b.key.devices().size()) {
      return true;
    }
    if (b.key.devices().size() > a.key.devices().size()) {
      return false;
    }

    // Prefer cliques with smaller id (comes earlier in execution order).
    return a.id < b.id;
  });

  return cliques;
}

}  // namespace xla::gpu
