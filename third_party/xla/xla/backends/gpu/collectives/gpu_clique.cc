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

#include "xla/backends/gpu/collectives/gpu_clique.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/container/btree_map.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/clique.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla::gpu {

GpuClique::GpuClique(
    GpuCliqueKey clique_key, std::optional<CliqueId> clique_id,
    absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators)
    : Clique(std::move(communicators)),
      clique_key_(clique_key),
      clique_id_(clique_id) {}

}  // namespace xla::gpu
