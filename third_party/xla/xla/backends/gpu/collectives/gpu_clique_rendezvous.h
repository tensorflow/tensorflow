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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_RENDEZVOUS_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_RENDEZVOUS_H_

#include <functional>
#include <memory>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/tsl/util/unique_any.h"
#include "xla/util.h"

namespace xla::gpu {

// A result of a rendezvous for a given clique key, where each participant is
// expected to join the rendezvous together with a local data that it wants to
// share with other participants. Supported only for local cliques.
class GpuCliqueRendezvous {
 public:
  // Joins a rendezvous for the given clique key and return the associated data.
  //
  // This is a collective operation that must be called concurrently by all
  // participating devices in the clique. Implementation relies on the
  // rendezvous synchronization to ensure that all ranks arrive to the barrier.
  // The result is collectively owned by all participants.
  static absl::StatusOr<std::shared_ptr<GpuCliqueRendezvous>> Join(
      const GpuCliqueKey& clique_key, RankId rank, tsl::UniqueAny data);

  // Returns the clique key associated with this rendezvous object.
  const GpuCliqueKey& clique_key() const { return clique_key_; }

  // Returns the value at the given rank. If value type is not the same as `T`,
  // returns an error.
  template <typename T>
  absl::StatusOr<std::reference_wrapper<const T>> at(RankId rank) const {
    auto it = values_.find(rank);
    if (it == values_.end()) {
      return NotFound("Data not found for rank %d", rank.value());
    }

    const T* ptr = tsl::any_cast<T>(&it->second);
    if (ptr == nullptr) {
      return InvalidArgument("Data type mismatch for rank %d", rank.value());
    }

    return std::reference_wrapper<const T>(*ptr);
  }

  // Returns a mutable reference to the value at the given rank.
  template <typename T>
  absl::StatusOr<std::reference_wrapper<T>> at(RankId rank) {
    auto it = values_.find(rank);
    if (it == values_.end()) {
      return NotFound("Data not found for rank %d", rank.value());
    }

    T* ptr = tsl::any_cast<T>(&it->second);
    if (ptr == nullptr) {
      return InvalidArgument("Data type mismatch for rank %d", rank.value());
    }

    return std::reference_wrapper<T>(*ptr);
  }

 private:
  GpuCliqueRendezvous(GpuCliqueKey clique_key,
                      absl::btree_map<RankId, tsl::UniqueAny> values);

  GpuCliqueKey clique_key_;
  absl::btree_map<RankId, tsl::UniqueAny> values_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_RENDEZVOUS_H_
