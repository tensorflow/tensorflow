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

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/container/btree_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/clique.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/lockable.h"

namespace xla::gpu {

class LockableGpuClique;

// A group of GPU communicators making up a clique for a given clique key.
class GpuClique : public Clique {
 public:
  GpuClique(
      GpuCliqueKey key, std::optional<CliqueId> id,
      absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators);

  // Returns true if clique is local: all communicators belong to current
  // process. Non-local cliques spans multiple processes (typically hosts).
  bool IsLocal() const { return num_communicators() == key_.devices().size(); }

  const GpuCliqueKey& key() const { return key_; }
  const std::optional<CliqueId>& id() const { return id_; }

  std::string DebugString() const final;
  absl::Status HealthCheck() const final;

 private:
  friend LockableGpuClique;

  // A functor to give human-readable names to lockable GPU cliques.
  struct LockableName {
    static std::string ToString(const GpuClique& clique);
  };

  GpuCliqueKey key_;
  std::optional<CliqueId> id_;
};

// A lockable version of GpuClique that guarantees exclusive access to the
// clique communicators.
class LockableGpuClique : public Lockable<GpuClique, GpuClique::LockableName> {
 public:
  LockableGpuClique(
      GpuCliqueKey clique_key, std::optional<CliqueId> clique_id,
      absl::btree_map<RankId, std::unique_ptr<Communicator>> communicators);

  std::string DebugString() const;

  // Checks for async errors for all the communicators in the clique without
  // taking the lock. If at least one of the communicators has an async error,
  // it returns one of the errors.
  absl::Status HealthCheck() const;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_GPU_CLIQUE_H_
