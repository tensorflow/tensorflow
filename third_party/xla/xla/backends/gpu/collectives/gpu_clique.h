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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_H_

#include <memory>
#include <optional>

#include "absl/container/btree_map.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/clique.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"

namespace xla::gpu {

// A group of GPU communicators making up a clique for a given clique key.
class GpuClique : public Clique {
 public:
  GpuClique(
      GpuCliqueKey clique_key, std::optional<CliqueId> clique_id,
      absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators);

  // Returns true if clique is local: all communicators belong to current
  // process. Non-local cliques spans multiple processes (typically hosts).
  bool IsLocal() const {
    return num_communicators() == clique_key_.devices().size();
  }

  const GpuCliqueKey& clique_key() const { return clique_key_; }
  const std::optional<CliqueId>& clique_id() const { return clique_id_; }

 private:
  GpuCliqueKey clique_key_;
  std::optional<CliqueId> clique_id_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_H_
