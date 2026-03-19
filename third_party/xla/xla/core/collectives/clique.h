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

#ifndef XLA_CORE_COLLECTIVES_CLIQUE_H_
#define XLA_CORE_COLLECTIVES_CLIQUE_H_

// A group of NCCL communicators making up a clique. With NCCL it's notoriously
// easy to get a deadlock, so we take extra care by grouping communicators into
// cliques and making sure that we have a well defined order of all collective
// operations that does not lead to deadlocks.

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla {

// A group of collective communicators for make up a clique.
//
// We use clique mechanism to group communicators to be able to efficiently
// get exclusive access to all communicators in a clique, as we typically have
// to guarantee that collective operations on all ranks are executed in the
// same order across all devices.
class Clique {
 public:
  explicit Clique(
      absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators);
  virtual ~Clique() = default;

  // Returns a communicator for a given rank if it's in a clique.
  std::optional<Communicator*> comm(RankId rank) const;

  // Adds a communicator to the clique.
  absl::Status AddComm(RankId rank, std::unique_ptr<Communicator> communicator);

  // Calls `fn` for each communicator in the clique.
  void ForEachComm(absl::FunctionRef<void(RankId, Communicator*)> fn) const;

  // Checks that all communicators in the clique are in a healthy state.
  virtual absl::Status HealthCheck() const = 0;

  // Returns a human-readable string representation of the clique.
  virtual std::string DebugString() const = 0;

  size_t num_communicators() const { return communicators_.size(); }

 private:
  // We keep communicators in a sorted order by rank to guarantee
  // deterministic traversal order in `ForEachComm`.
  absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators_;
};

}  // namespace xla

#endif  // XLA_CORE_COLLECTIVES_CLIQUE_H_
