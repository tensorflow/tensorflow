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

#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/runtime/device_id.h"
#include "xla/util.h"

namespace xla::gpu {

absl::Status CollectiveCliqueRequests::RequestClique(
    const GpuCliqueKey& clique_key,
    absl::Span<const std::vector<GlobalDeviceId>> device_groups,
    const CliqueRequirements& requirements) {
  // Each replica group should have devices in sorted order.
  DCHECK(absl::c_all_of(device_groups, [](const auto& g) {
    return absl::c_is_sorted(g);
  })) << "Device group must be sorted in ascending order";

  // And replica groups themselves should be sorted (lexicographic order on
  // sorted inner groups is equivalent to sorting by first device id).
  DCHECK(absl::c_is_sorted(device_groups))
      << "Device groups must be sorted by first device id";

  VLOG(5) << absl::StreamFormat(
      "Add collective clique request: %v; device_groups=[%s]; "
      "requirements=%v",
      clique_key, HumanReadableDeviceGroups(device_groups), requirements);

  // If the clique already exists, update it with new requirements.
  if (auto it = cliques_.find(clique_key); it != cliques_.end()) {
    CliqueRequest& req = it->second;

    // It is illegal to request the same GPU clique with different device
    // groups. This must never happen under SPMD programming model.
    if (!absl::c_equal(req.device_groups, device_groups)) {
      return InvalidArgument(
          "GPU clique %v requested from different device groups: [%s] vs [%s]",
          clique_key, HumanReadableDeviceGroups(req.device_groups),
          HumanReadableDeviceGroups(device_groups));
    }

    if (requirements.dev_comm) {
      req.dev_comms.insert(*requirements.dev_comm);
    }

    if (requirements.barrier_reqs.has_value()) {
      req.barrier_after_module_execution_requested |=
          requirements.barrier_reqs->module_execution_barrier;
    }

    return absl::OkStatus();
  }

  // XLA compiler guarantees that all collective operations have the same
  // order on all replicas. We rely on this property to assign unique id to
  // clique requests simply based on the number of already recorded requests.
  CliqueRequest req{/*id=*/cliques_.size(),
                    clique_key,
                    {device_groups.begin(), device_groups.end()}};
  if (requirements.dev_comm) {
    req.dev_comms.insert(*requirements.dev_comm);
  }

  if (requirements.barrier_reqs.has_value()) {
    req.barrier_after_module_execution_requested |=
        requirements.barrier_reqs->module_execution_barrier;
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

absl::flat_hash_set<GlobalDeviceId>
CollectiveCliqueRequests::GetDevicesRequiringBarrier() const {
  absl::flat_hash_set<GlobalDeviceId> result;
  for (const auto& [key, request] : cliques_) {
    if (!request.barrier_after_module_execution_requested) {
      continue;
    }

    for (const std::vector<GlobalDeviceId>& group : request.device_groups) {
      for (GlobalDeviceId device : group) {
        result.insert(device);
      }
    }
  }
  return result;
}

}  // namespace xla::gpu
