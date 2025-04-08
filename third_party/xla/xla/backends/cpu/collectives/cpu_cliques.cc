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

#include "xla/backends/cpu/collectives/cpu_cliques.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/cpu/collectives/cpu_clique.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::cpu {

//===----------------------------------------------------------------------===//
// ProcessCpuCliques
//===----------------------------------------------------------------------===//

namespace {

// CpuClique is not thread-safe, so we wrap it in a thread-safe container as we
// create new communicators lazily and potentially from multiple threads.
struct ThreadSafeClique {
  explicit ThreadSafeClique(CpuCliqueKey key) : clique(key) {}

  absl::Mutex mu;
  CpuClique clique ABSL_GUARDED_BY(mu);
};

// Container for initialized and ready to use CPU cliques. In contrast to GPU
// cliques, CPU cliques are not lockable, and we create communicators lazily
// when needed.
struct ProcessCpuCliques {
  using Key = std::pair<CpuCollectives*, CpuCliqueKey>;

  absl::Mutex mu;
  absl::node_hash_map<Key, ThreadSafeClique> map ABSL_GUARDED_BY(mu);
};

}  // namespace

// Returns process-local CPU cliques.
static ProcessCpuCliques& GetProcessCpuCliques() {
  static auto* cliques = new ProcessCpuCliques;
  return *cliques;
}

// Erases cliques constructed from a given instance of CpuCollectives.
static void EraseProcessCpuCliques(CpuCollectives* collectives) {
  VLOG(3) << "Erase process CPU cliques for collectives: " << collectives;
  ProcessCpuCliques& cliques = GetProcessCpuCliques();

  absl::MutexLock lock(&cliques.mu);
  absl::erase_if(cliques.map, [collectives](const auto& entry) {
    return entry.first.first == collectives;
  });
}

//===----------------------------------------------------------------------===//

// TODO(b/380457503): Consider switching to a lockable CPU clique model similar
// to GPU cliques, and creating all communicators upfront.
absl::StatusOr<Communicator*> AcquireCommunicator(
    CpuCollectives* collectives, const CpuCliqueKey& clique_key, RankId rank) {
  VLOG(3) << "Acquire communicator for clique key " << clique_key.ToString()
          << " and rank " << rank << " from collectives: " << collectives;

  ProcessCpuCliques& cliques = GetProcessCpuCliques();

  // Synchronize access to the process cliques.
  ThreadSafeClique& thread_safe_clique = [&]() -> ThreadSafeClique& {
    absl::MutexLock lock(&cliques.mu);
    auto [it, emplaced] = cliques.map.try_emplace(
        std::make_pair(collectives, clique_key), clique_key);

    // If we created a new clique, register a callback to erase it when the
    // collectives instance is destroyed.
    if (emplaced) {
      VLOG(3) << "Created a new clique for clique key " << clique_key.ToString()
              << " and collectives: " << collectives;
      collectives->AddOnDestroyCallback(
          [collectives] { EraseProcessCpuCliques(collectives); });
    }

    return it->second;
  }();

  // Check if we already have a communicator for this rank.
  std::optional<Communicator*> comm = [&]() -> std::optional<Communicator*> {
    absl::MutexLock lock(&thread_safe_clique.mu);
    return thread_safe_clique.clique.comm(rank);
  }();

  if (comm.has_value()) return *comm;

  VLOG(3) << "Create a new communicator for clique key "
          << clique_key.ToString() << " and rank " << rank;

  // Create a new communicator and add it to the clique.
  CpuCollectives::DeviceRank device_rank(/*device=*/nullptr, rank);
  CpuCollectives::Config config;

  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Communicator>> communicators,
                      collectives->CreateCommunicators(clique_key, std::nullopt,
                                                       {device_rank}, config));

  // We expect to create communicators lazily on at a time.
  if (communicators.size() != 1) {
    return Internal(
        "Expected to create a single communicator for a clique key %s and rank "
        "%d, but got %d",
        clique_key.ToString(), rank.value(), communicators.size());
  }

  absl::MutexLock lock(&thread_safe_clique.mu);

  // Check if we lost a race to create a communicator to another thread.
  if (auto comm = thread_safe_clique.clique.comm(rank)) {
    return *comm;
  }

  TF_RETURN_IF_ERROR(thread_safe_clique.clique.AddComm(
      rank, std::move(communicators.front())));

  return *thread_safe_clique.clique.comm(rank);
}

}  // namespace xla::cpu
