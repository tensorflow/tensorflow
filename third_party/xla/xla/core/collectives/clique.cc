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

#include "xla/core/collectives/clique.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/functional/function_ref.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla {

Clique::Clique(
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators)
    : communicators_(std::move(communicators)) {}

std::optional<Communicator*> Clique::comm(RankId rank) const {
  if (auto it = communicators_.find(rank); it != communicators_.end()) {
    return it->second.get();
  }
  return std::nullopt;
}

void Clique::ForEachComm(
    absl::FunctionRef<void(RankId, Communicator*)> fn) const {
  for (auto& [rank, comm] : communicators_) {
    fn(rank, comm.get());
  }
}

}  // namespace xla
