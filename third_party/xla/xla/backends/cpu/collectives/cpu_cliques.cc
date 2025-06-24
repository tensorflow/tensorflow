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

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
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

  // Used to construct every communicator once.
  absl::flat_hash_map<RankId, std::unique_ptr<absl::once_flag>> create_comm_once
      ABSL_GUARDED_BY(mu);

  // The status of constructing a communicator with a particular rank.
  absl::flat_hash_map<RankId, absl::Status> create_comm_status
      ABSL_GUARDED_BY(mu);
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

absl::StatusOr<std::unique_ptr<Communicator>> CreateCommunicator(
    CpuCollectives* collectives, const CpuCliqueKey& clique_key, RankId rank) {
  VLOG(3) << "Create a new communicator for clique key "
          << clique_key.ToString() << " and rank " << rank;

  CpuCollectives::DeviceRank device_rank(/*device=*/nullptr, rank);
  CpuCollectives::Config config;
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<Communicator>> communicators,
                      collectives->CreateCommunicators(clique_key, std::nullopt,
                                                       {device_rank}, config));

  if (communicators.size() != 1) {
    // We expect to create communicators lazily, one at a time.
    return Internal(
        "Expected to create a single communicator for a clique key %s and "
        "rank %d, but got %d",
        clique_key.ToString(), rank.value(), communicators.size());
  }

  return std::move(communicators.front());
};

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

  // Create the communicator, once.
  absl::once_flag* create_comm_once = nullptr;
  {
    absl::MutexLock lock(&thread_safe_clique.mu);
    std::unique_ptr<absl::once_flag>& x =
        thread_safe_clique.create_comm_once[rank];
    if (!x) {
      x = std::make_unique<absl::once_flag>();
    }
    create_comm_once = x.get();
  }
  absl::call_once(*create_comm_once, [&]() {
    absl::StatusOr<std::unique_ptr<Communicator>> comm =
        CreateCommunicator(collectives, clique_key, rank);

    absl::MutexLock lock(&thread_safe_clique.mu);
    if (!comm.ok()) {
      thread_safe_clique.create_comm_status[rank] = comm.status();
      return;
    }
    absl::Status s = thread_safe_clique.clique.AddComm(rank, *std::move(comm));
    if (!s.ok()) {
      thread_safe_clique.create_comm_status[rank] = s;
    }
  });

  absl::MutexLock lock(&thread_safe_clique.mu);
  TF_RETURN_IF_ERROR(thread_safe_clique.create_comm_status[rank]);
  return *thread_safe_clique.clique.comm(rank);
}

}  // namespace xla::cpu
