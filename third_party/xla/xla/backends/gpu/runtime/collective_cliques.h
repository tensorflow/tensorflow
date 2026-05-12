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

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_communicator.h"
#include "xla/backends/gpu/runtime/collective_clique_requests.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/service/rendezvous.h"
#include "xla/tsl/util/tied_ref.h"
#include "xla/util.h"

namespace xla::gpu {

// A collection of collective cliques acquired based on GPU clique requests
// collected from all thunks at prepare stage.
class CollectiveCliques {
 public:
  CollectiveCliques() = default;
  explicit CollectiveCliques(AcquiredCliquesMap cliques_map);

  absl::StatusOr<GpuCommunicator*> GetComm(const GpuCliqueKey& clique_key,
                                           RankId rank) const;

  absl::StatusOr<GpuCommunicator*> GetComm(
      const GpuCliqueKey& clique_key, GlobalDeviceId global_device_id) const;

  absl::StatusOr<GpuDeviceCommunicator*> GetDeviceComm(
      const GpuCliqueKey& clique_key, RankId rank,
      const GpuDeviceCommunicator::Requirements& reqs) const;

  absl::StatusOr<GpuDeviceCommunicator*> GetDeviceComm(
      const GpuCliqueKey& clique_key, GlobalDeviceId global_device_id,
      const GpuDeviceCommunicator::Requirements& reqs) const;

  // Returns whether peer device memory access is possible between all devices
  // in the clique.
  absl::StatusOr<bool> peer_access_enabled(
      const GpuCliqueKey& clique_key) const;

  // Ties an object to a clique. Clique takes ownership of the object and will
  // destroy it when the clique is destroyed. When TiedRef is destroyed, the
  // object will be garbage collected.
  template <typename T>
  absl::StatusOr<tsl::TiedRef<T>> Tie(const GpuCliqueKey& clique_key,
                                      std::unique_ptr<T> object);

  bool empty() const { return cliques_map_.empty(); }

  absl::StatusOr<std::pair<RendezvousFlag*, RendezvousFlag*>>
  GetCliqueFirstRendezvousFlags(const GpuCliqueKey& clique_key,
                                absl::string_view module_name) const;

 private:
  AcquiredCliquesMap cliques_map_;
};

template <typename T>
absl::StatusOr<tsl::TiedRef<T>> CollectiveCliques::Tie(
    const GpuCliqueKey& clique_key, std::unique_ptr<T> object) {
  // Check that we locked access to a clique for `clique_key`.
  auto clique = cliques_map_.find(clique_key);
  if (clique == cliques_map_.end()) {
    return NotFound("No clique found for clique key: %v", clique_key);
  }
  return (*clique->second)->Tie(std::move(object));
}

// Acquires collective cliques using the given collective parameters for all
// requested GPU cliques.
//
// WARNING: This is a collective operation, that must be called by all
// participating ranks in the requested cliques, otherwise it will lead to a
// deadlock.
absl::StatusOr<CollectiveCliques> AcquireCollectiveCliques(
    const CollectiveParams& params, const CollectiveCliqueRequests& cliques);

absl::StatusOr<bool> AllFirstRendezvousCompleted(
    const CollectiveCliques& collective_cliques,
    const std::vector<GpuCliqueKey>& requested_clique_keys,
    absl::string_view module_name);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_CLIQUES_H_
